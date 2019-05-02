# Download the data
if [ -d "checkpoints/" ]; then
    echo "The data had been already downloaded"
else
    mkdir checkpoints/

    curl -L https://www.dropbox.com/s/lybghg3ne2quaw8/memory.pth?dl=1 > checkpoints/memory.pth
    curl -L https://www.dropbox.com/s/6ncv1zy54lppbsl/memory_chinese.pth?dl=1 > checkpoints/memory_chinese.pth
    curl -L https://www.dropbox.com/s/vwwnyfv404j0yhq/data.bin?dl=1 > data/data.bin
    curl -L https://www.dropbox.com/s/hf5rnmw055ogoe8/chinese_data.bin?dl=1 > data/chinese_data.bin
    curl -L https://www.dropbox.com/s/kn1ol7ilb8dx2cc/xiaohuangji50w_fenciA.conv?dl=1 > data/xiaohuangji50w_fenciA.conv
fi

wait

if [ $1 == "train" ]; then
    if [ $2 == "chinese" ]; then
        python3.6 python main.py train --chinese=True
    elif [ $2 == "english" ]; then
        python main.py train --chinese=False
    fi
elif [ $1 == "test" ]; then
    if [ $2 == "chinese" ]; then
        python main.py test --chinese=True
    elif [ $2 == "english" ]; then
        python main.py test --chinese=False
    fi
fi

