#!/bin/bash
printenv | grep JOBMONITOR | sudo tee - /etc/default/telegraf
sudo service telegraf start
exec "$@"
