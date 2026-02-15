#!/bin/bash
# Launch Chrome with sandbox profile for CDP testing
# Remote debugging on port 9220
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --user-data-dir=/Users/john/.config/sfb/chrome_profiles/sandbox \
  --remote-debugging-port=9220 \
  --remote-allow-origins=* \
  --no-first-run \
  --no-default-browser-check \
  "$@" &
