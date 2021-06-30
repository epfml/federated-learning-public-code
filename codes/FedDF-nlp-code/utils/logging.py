# -*- coding: utf-8 -*-
import os
import json
import time
import platform


class Logger:
    """
    Very simple prototype logger that will store the values to a JSON file
    """

    def __init__(self, file_folder):
        """
        :param filename: ending with .json
        :param auto_save: save the JSON file after every addition
        """
        self.file_folder = file_folder
        self.file_json = os.path.join(file_folder, "log-1.json")
        self.file_txt = os.path.join(file_folder, "log.txt")
        self.values = []

    def log_metric(self, name, values, tags, display=False):
        """
        Store a scalar metric

        :param name: measurement, like 'accuracy'
        :param values: dictionary, like { epoch: 3, value: 0.23 }
        :param tags: dictionary, like { split: train }
        """
        self.values.append({"measurement": name, **values, **tags})

        if display:
            float_keys = [k for k, v in values.items() if type(v) == float]
            for k in float_keys:
                values[k] = round(values[k], 5)
            print(
                "{name}: {values} ({tags})".format(name=name, values=values, tags=tags)
            )

    def log(self, value):
        content = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + value
        print(content)
        self.save_txt(content)

    def save_json(self):
        """Save the internal memory to a file."""
        with open(self.file_json, "w") as fp:
            json.dump(self.values, fp, indent=" ")

        if len(self.values) > 1e3:
            # reset 'values' and redirect the json file to other name.
            self.values = []
            self.redirect_new_json()

    def save_txt(self, value):
        write_txt(value + "\n", self.file_txt, type="a")

    def redirect_new_json(self):
        """get the number of existing json files under the current folder."""
        existing_json_files = [
            file
            for file in os.listdir(self.file_folder)
            if "json" in file and "log" in file
        ]
        self.file_json = os.path.join(
            self.file_folder, "log-{}.json".format(len(existing_json_files) + 1)
        )


def display_args(conf):
    print("\n\nparameters: ")
    for arg in vars(conf):
        print("\t" + str(arg) + "\t" + str(getattr(conf, arg)))

    print(f"\n\nexperiment platform: {platform.node()} with GPUs {conf.world}")
    save_arguments(conf)


def save_arguments(conf):
    with open(os.path.join(conf.checkpoint_root, "arguments.json"), "w") as fp:
        safe_serialize(conf.__dict__, fp)


def safe_serialize(obj, fp):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dump(obj, fp, default=default, indent=" ")


def write_txt(data, out_path, type="w"):
    """write the data to the txt file."""
    with open(out_path, type) as f:
        f.write(data)
