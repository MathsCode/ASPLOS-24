if [ "$1" == "EE" ]; then 
CUDA_VISIBLE_DEVICES="$2" python AutoEE/EEinference.py > ./results/EEinference.txt
fi
if [ "$1" == "baseline" ]; then 
CUDA_VISIBLE_DEVICES="$2" python baseline/TD_inference.py > ./results/baseline.txt
fi