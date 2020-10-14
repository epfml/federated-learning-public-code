FROM user/base

USER root

# Install telegraf
RUN wget https://dl.influxdata.com/telegraf/releases/telegraf_1.8.3-1_amd64.deb -O telegraf.deb \
    && dpkg -i telegraf.deb \
    && rm telegraf.deb
COPY base-with-db/telegraf.conf /etc/telegraf/telegraf.conf

# Required environment variables:
# Telegraf
ENV JOBMONITOR_TELEGRAF_HOST='localhost' \
    JOBMONITOR_TELEGRAF_PORT='8092'

# InfluxDB
ENV JOBMONITOR_TIMESERIES_HOST='' \
    JOBMONITOR_TIMESERIES_PORT='8086' \
    JOBMONITOR_TIMESERIES_DB='jobmonitor'

# MongoDB
ENV JOBMONITOR_METADATA_HOST='' \
    JOBMONITOR_METADATA_PORT='27017' \
    JOBMONITOR_METADATA_DB='jobmonitor'

# Add an entrypoint script that will start the telegraf service
COPY base-with-db/entrypoint.sh /entrypoint.sh
RUN chmod 777 /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
