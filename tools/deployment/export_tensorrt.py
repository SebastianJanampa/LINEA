import os 

input_folder = 'onnx_engines'
input_files = [f for f in os.listdir(input_folder)]

output_folder = 'trt_engines'
output_files = [f.replace('onnx', 'engine') for f in input_files]

os.makedirs(output_folder, exist_ok=True)

trtexec="/usr/src/tensorrt/bin/trtexec"

for f_in, f_out in zip(input_files, output_files):
	cmd = f'{trtexec} --onnx="{input_folder}/{f_in}" --saveEngine="{output_folder}/{f_out}" --fp16'
	print(f'running:\t{cmd}')
	os.system(cmd)