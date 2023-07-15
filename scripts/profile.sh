#!/bin/sh

OUTPUT_DIR=results/profile/

poetry run scalene --reduced-profile -m app \
&& mkdir -p $OUTPUT_DIR \
&& mv profile.html $OUTPUT_DIR \
&& mv profile.json $OUTPUT_DIR \
&
