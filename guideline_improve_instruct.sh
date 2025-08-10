ls /usr/local/ | grep cuda
### If it was any version else
sudo rm -r /usr/local/cuda-12.1/
sudo apt clean && sudo apt autoclean
### Install new CUDA version
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

cd get_data
# python do_inference.py --model_path google/gemma-3-4b-it
CUDA_VISIBLE_DEVICES=1 python get_data/do_inference.py --data_path=get_data/data/data.jsonl --res_path=get_data/data/res_gemma3.jsonl --model_path=google/gemma-3-4b-it
python check.py --input_data=data/data.jsonl --input_response_data=data/res_gemma-3-4b-it.jsonl --output_dir=results --output_file_name=check_gemma-3-4b-it
