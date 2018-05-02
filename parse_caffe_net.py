import sys
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
sys.path.append("..")
import google.protobuf as pb
import string
import argparse

def string_remove(input_str):
    input_str = input_str.replace("(","")
    input_str = input_str.replace(")","")
    # TODO with regex
    #import re
    #blob_str = re.sub(r"\(\)","",blob_str)
    return input_str
    
def parse_model_stat(net, model):   
    file_name = "model_blob_stat.csv"
    model_stat=open(file_name,'w+')
    model_stat.write("header for the cnn \n");
    stat_line=[]
    print('******************************blob info************************************')
    for layer, blob in net.blobs.iteritems():
        blob_str = layer+','+ str(blob.data.shape)
        blob_str = string_remove(blob_str)
        print(blob_str)
        #add the convolution parameters
        if "Conv" in layer:
            param = net.params[layer]
            blob_str += ","
            shape = string_remove(str(param[0].data.shape))
            blob_str += shape
            #add the pad stride kernel
            for cl in model.layer:
                if cl.name == layer:
                    kernel = cl.convolution_param.kernel_size[0] if len(cl.convolution_param.kernel_size) else 1
                    stride = cl.convolution_param.stride[0] if len(cl.convolution_param.stride) else 1
                    pad = cl.convolution_param.pad[0] if len(cl.convolution_param.pad) else 0
                    # string join
                    conv_param_line =","+str(kernel)+"," + str(stride) + "," + str(pad)
                    blob_str += conv_param_line
        if "conv" in layer:
            param = net.params[layer]
            blob_str += ","
            shape = string_remove(str(param[0].data.shape))
            blob_str += shape
        blob_str+="\n"
        stat_line.append(blob_str)
    model_stat.writelines(stat_line)
    print('*****************************param info************************************')
    file_name = "model_param_stat.csv"
    model_stat=open(file_name,'w+')
    model_stat.write("header for the cnn \n");
    stat_line=[]
    for layer, param in net.params.iteritems():
        parma_str = layer+','+ str(param[0].data.shape)+"\n"
        parma_str = string_remove(parma_str)
        print(parma_str)
        stat_line.append(parma_str)
    model_stat.writelines(stat_line)
    print('*****************************model info************************************')
    file_name = "model_text_stat.csv"
    model_stat=open(file_name,'w+')
    model_stat.write("header for the cnn \n");
    for cl in model.layer:
        text = str(cl.type)+":" +str(cl.name) +"\n"
        if cl.type == "Convolution":
            kernel = cl.convolution_param.kernel_size[0] if len(cl.convolution_param.kernel_size) else 1
            stride = cl.convolution_param.stride[0] if len(cl.convolution_param.stride) else 1
            pad = cl.convolution_param.pad[0] if len(cl.convolution_param.pad) else 0
            #output = cl.output_param[0].out_channel
            print(str(kernel)+" " + str(stride) + " " + str(pad))
        print(text)
    print('******************************end info*************************************')


def run(prototxt,caffemodel):
	model = caffe_pb2.NetParameter()
	with open(caffemodel,'rb') as f:
		model.ParseFromString(f.read())

	net = caffe.Net(prototxt,caffe.TEST,weights = caffemodel)
	parse_model_stat(net,model)

def parse_args():
	parser = argparse.ArgumentParser('Model stat')
	parser.add_argument('-model',dest = 'model',help = 'path to the prototxt file',type=str)
	parser.add_argument('-weight', dest = 'weights', help ='path to the weight file',type = str)
	return parser.parse_args()

if __name__ == '__main__':

	args = parse_args()	
	run(args.model,args.weights)

	print("complelte the parsing")

