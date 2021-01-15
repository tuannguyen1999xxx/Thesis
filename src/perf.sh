#!/bin/bash
perf_client -i gRPC -u localhost:8001 -m retinaface --percentile=95 --concurrency-range 1:8 -v -a -b 8