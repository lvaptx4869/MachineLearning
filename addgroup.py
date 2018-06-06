#!/usr/bin/env python
"""
Convert the Depthwise Convolution prototxt to pure Convolution prototxt.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2
import argparse
from google.protobuf.text_format import  Merge

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('src_prototxt')
    parser.add_argument('dst_prototxt')

    args = parser.parse_args()
    net = caffe_pb2.NetParameter()
    Merge(open(args.src_prototxt,'r').read(),net)
    for layer in net.layer:
        if layer.type =="ConvolutionDepthwise":
			layer.type ="Convolution"
			print layer.convolution_param.num_output
			print layer.convolution_param.group
			layer.convolution_param.group=layer.convolution_param.num_output
    with open(args.dst_prototxt,'w') as dst_file:
        dst_file.write(str(net))



if __name__ == '__main__':
    main()
