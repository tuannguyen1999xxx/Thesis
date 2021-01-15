import argparse
import numpy as np
import os
from builtins import range
from PIL import Image
# from functools import partial
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
from utils.utils import timer

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

class trtis_engine():
    def __init__(self, model_name, url, protocol, batch_size, is_async, is_streaming):
        self.model_name = model_name
        self.url = url
        self.batch_size = batch_size
        self.async_set = is_async
        self.streaming = is_streaming 
        self.verbose = False
        self.model_version = 1
        self.input_type = "FP32"
        self.protocol = ProtocolType.from_str(protocol)
        self.frame_id = 0
        

        if self.streaming and self.protocol != ProtocolType.GRPC:
            raise Exception("Streaming is only allowed with gRPC protocol")
        
        self.max_batch_size, self.input_names, self.output_names = self.get_info()

        # ctx_infer = InferContext(url, ProtocolType.from_str(protocol), model_name, \
        #                "", False, 0, is_streaming)
        self.ctx = InferContext(self.url, self.protocol, self.model_name,
                        self.model_version, self.verbose, 0, self.streaming)
    def completion_callback(self, input_filenames, user_data, infer_ctx, request_id):
        user_data._completed_requests.put((request_id, input_filenames))

    def get_info(self):
        ctx_status = ServerStatusContext(self.url, self.protocol, self.model_name, self.verbose)
        #TODO slow in get server status
        server_status = ctx_status.get_server_status()
        status = server_status.model_status[self.model_name]
        config = status.config
        inputs = []
        outputs = []
        # [print("input.name: ", input.name) for input in config.input]
        [inputs.append(input.name) for input in config.input]
        output = config.output[0]
        [outputs.append(output.name) for output in config.output]

        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if self.batch_size != 1:
                raise Exception("batching not supported for model '" + self.model_name + "'")
        else: # max_batch_size > 0
            if self.batch_size > max_batch_size:
                raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, self.model_name))
        
        return (max_batch_size, inputs, outputs)

    @timer
    def inferencing(self, image_data):
        
        results = []
        result_filenames = []
        request_ids = []
        image_idx = 0
        last_request = False
        
        self.frame_id += 1
        user_data = UserData()
        processed_count = 0
        sent_count = 0

        while not last_request:
            input_filenames = []
            input_batch = []
            for idx in range(self.batch_size):
                input_filenames.append(idx)
                input_batch.append(image_data[image_idx])

                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True

            # Send request
            if not self.async_set:
                results.append(self.ctx.run(
                    {   self.input_names[0] : input_batch },
                    {   self.output_names[0] : (InferContext.ResultFormat.RAW),
                        self.output_names[1] : (InferContext.ResultFormat.RAW),
                        self.output_names[2] : (InferContext.ResultFormat.RAW), 
                        self.output_names[3] : (InferContext.ResultFormat.RAW)},
                                self.batch_size))
                result_filenames.append(input_filenames)
                sent_count += 1
            else:
                # print("partial(self.completion_callback, self.frame_id, user_data): ", partial(completion_callback, self.frame_id, self.user_data))
                self.ctx.async_run(partial(self.completion_callback, self.frame_id, user_data),
                                {   self.input_names[0] : input_batch },
                                {   self.output_names[0] : (InferContext.ResultFormat.RAW),
                                    self.output_names[1] : (InferContext.ResultFormat.RAW),
                                    self.output_names[2] : (InferContext.ResultFormat.RAW),
                                    self.output_names[3] : (InferContext.ResultFormat.RAW) },
                                                        self.batch_size)
                sent_count += 1
        # For async, retrieve results according to the send order
        if self.async_set:
            # while not self.user_data._completed_requests.empty():
            while processed_count < sent_count:
                (request_id, frame_id) = user_data._completed_requests.get()
                results.append(self.ctx.get_async_run_results(request_id))
                result_filenames.append(frame_id)
                processed_count += 1
        
        return results, result_filenames, self.output_names