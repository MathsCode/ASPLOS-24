if [ "$1" == "EE" ]; then 
python AutoEE/EEinference.py > ./results/EEinference.txt
fi
if [ "$1" == "baseline" ]; then 
python baseline/TD_inference.py > ./results/baseline.txt
fi