# -*- coding: utf-8 -*-
import os
import socket
import datetime
from copy import deepcopy
from bson.objectid import ObjectId
from itertools import groupby

import numpy as np

log_metric_fn = None
mongo = None


"""some operators for mongodb."""


def init_mongo(conf):
    from pymongo import MongoClient

    mongo_client = MongoClient(
        host=os.getenv("JOBMONITOR_METADATA_HOST"),
        port=int(os.getenv("JOBMONITOR_METADATA_PORT")),
    )
    global mongo

    mongo = getattr(mongo_client, os.getenv("JOBMONITOR_METADATA_DB"))
    # init the job status at mongodb.
    job_content = {
        "user": conf.user,
        "project": conf.project,
        "experiment": conf.experiment,
        "config": get_clean_arguments(conf),
        "job_id": conf.timestamp,
        "rank_id": get_rank(conf),
        "host": socket.gethostname(),
        "status": "SCHEDULED",
        "schedule_time": datetime.datetime.utcnow(),
        "output_dir": get_checkpoint_dir(conf),
        "is_cuda": conf.on_cuda,
    }

    conf._mongo_job_id = mongo.job.insert_one(job_content)
    conf.mongo_job_id = {"_id": ObjectId(str(conf._mongo_job_id.inserted_id))}

    # set job to 'started' in MongoDB
    update_mongo_record(
        conf.mongo_job_id,
        {"$set": {"status": "RUNNING", "start_time": datetime.datetime.utcnow()}},
    )


def announce_job_termination_to_mongo(conf):
    global mongo
    end_time = datetime.datetime.utcnow()

    # update.
    mongo.job.update(
        conf.mongo_job_id,
        {
            "$set": {
                "status": "FINISHED",
                "end_time": end_time,
                "lasted_time": (
                    end_time
                    - find_record_from_mongo(conf.mongo_job_id)[0]["start_time"]
                ).seconds,
            }
        },
    )


def update_mongo_record(mongo_job_id, content):
    global mongo
    mongo.job.update_one(mongo_job_id, content)


def find_mongo_record(mongo_job_id):
    global mongo
    return mongo.job.find_one(mongo_job_id)


def delete_mongo_record(mongo_job_id):
    global mongo
    return mongo.job.delete_one(mongo_job_id)


def delete_mongo_collection():
    global mongo
    mongo.job.drop()


def find_record_from_mongo(condition, projection=None):
    # some examples.
    # db.find(projection={"pmc_id": True})
    # db.find({"pmc_id": {"$ne": ""}})
    global mongo
    return [s for s in mongo.job.find(condition, projection=projection)]


def _get_non_duplicated_time(records):
    def _get_used_gpu_ids(record):
        if "conf" in record:
            conf = record["conf"]
            gpus = conf["world"].split(",")[: conf["blocks"]]
            return set(int(gpu) for gpu in gpus)
        else:
            return set([0])

    def _organize_results_per_host(_records):
        # preprocessing.
        used_gpus = set()
        list_of_start_time = []
        list_of_end_time = []

        for record in _records:
            # get used gpus.
            record["gpus"] = _get_used_gpu_ids(record)
            used_gpus.update(record["gpus"])

            # drop the microsecond
            record["start_time"] = record["start_time"].replace(microsecond=0)
            record["end_time"] = record["end_time"].replace(microsecond=0)
            list_of_start_time.append(record["start_time"])
            list_of_end_time.append(record["end_time"])

        # build time matrix.
        num_gpus = len(used_gpus)
        start_time = min(list_of_start_time)
        end_time = max(list_of_end_time)
        time_steps = int((end_time - start_time).total_seconds())
        time_matrix = np.zeros((num_gpus, time_steps))

        # fill in the time matrix.
        for record in _records:
            for gpu in record["gpus"]:
                start_time_idx = int(
                    (record["start_time"] - start_time).total_seconds()
                )
                end_time_idx = int((record["end_time"] - start_time).total_seconds())
                time_matrix[gpu, list(range(start_time_idx, end_time_idx))] = 1

        # merge results
        return time_matrix.sum()

    # sort records.
    new_records = []
    records = sorted(records, key=lambda x: x["host"])

    for _, values in groupby(records, key=lambda x: x["host"]):
        new_records += [_organize_results_per_host(list(values))]
    return sum(new_records)


def get_gpu_hours_from_mongo(year, month, day):
    # init client.
    from pymongo import MongoClient

    mongo_client = MongoClient(
        host=os.getenv("JOBMONITOR_METADATA_HOST"),
        port=int(os.getenv("JOBMONITOR_METADATA_PORT")),
    )

    mongo = getattr(mongo_client, os.getenv("JOBMONITOR_METADATA_DB"))

    # define the time range.
    end_time = datetime.datetime(year, month, day, 23, 59, 59)
    start_time = end_time - datetime.timedelta(days=7)

    # get all GPU hours.
    matched_records = [
        s
        for s in mongo.job.find(
            {
                "is_cuda": True,
                "status": "FINISHED",
                "start_time": {"$gt": start_time, "$lt": end_time},
            }
        )
    ]
    return 1.0 * _get_non_duplicated_time(matched_records) / 60 / 60


def get_clean_arguments(conf):
    copy_conf = deepcopy(conf)

    if "graph" in conf:
        copy_conf._graph = conf.graph.__dict__
        copy_conf.graph = None
    return copy_conf.__dict__


def get_rank(args):
    return args.graph.rank if "graph" in args else "root"


def get_checkpoint_dir(args):
    return args.checkpoint_root if "checkpoint_root" in args else ""


"""some operators for telegraf."""


def init_telegraf(args):
    from telegraf.client import TelegrafClient

    telegraf_client = TelegrafClient(
        host=os.getenv("JOBMONITOR_TELEGRAF_HOST"),
        port=int(os.getenv("JOBMONITOR_TELEGRAF_PORT")),
        tags={
            "host": socket.gethostname(),
            "user": args.user,
            "project": args.project,
            "experiment": args.experiment,
            "job_id": args.timestamp,
            "job_details": args.job_details,
            "job_info": args.job_info,
        },
    )

    global log_metric_fn
    log_metric_fn = telegraf_client.metric


def log_metric(*args):
    return log_metric_fn(*args)


"""some operators for influxdb."""


def init_influxdb(db_name="jobmonitor"):
    from influxdb import InfluxDBClient

    influx_client = InfluxDBClient(
        host=os.getenv("JOBMONITOR_TIMESERIES_HOST"), database=db_name
    )
    return influx_client


def get_measurement(cli, measurement=None, tags={}):
    rs = cli.query("select * from {}".format(measurement))
    return list(rs.get_points(measurement=measurement, tags=tags))
