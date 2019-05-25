# Download the data
if [ -d "checkpoints/" ]; then
    echo "The data had been already downloaded"
else
    
    mkdir checkpoints/
    
    # english mode data
    curl -L https://www.dropbox.com/s/lybghg3ne2quaw8/memory.pth?dl=1 > checkpoints/memory.pth
    curl -L https://www.dropbox.com/s/g99h2e68ei1k7rt/memory_new.pth?dl=1 > checkpoints/memory_new.pth
    curl -L https://www.dropbox.com/s/xp6btg9wbf5hpkf/memory_rl.pth?dl=1 > checkpoints/memory_rl.pth
    curl -L https://www.dropbox.com/s/vwwnyfv404j0yhq/data.bin?dl=1 > data/data.bin
    curl -L https://www.dropbox.com/s/ti107m6slp1k3ud/data_new.bin?dl=1 > data/data_new.bin
    
    # chinese chickbot mode data
    curl -L https://www.dropbox.com/s/lyrc1geco7kr70x/memory_chinese.pth?dl=1 > checkpoints/memory_chinese.pth
    curl -L https://www.dropbox.com/s/2jrm4dl5dijpo1w/chinese_data.bin?dl=1 > data/chinese_data.bin

    # chinese YourFBbot mode data
    curl -L https://www.dropbox.com/s/ign6qvvxyrfbl36/memory_fb.pth?dl=1 > checkpoints/memory_fb.pth
    curl -L https://www.dropbox.com/s/5z11n0xsabk0skr/fb_data.bin?dl=1 > data/fb_data.bin
    curl -L https://www.dropbox.com/s/a4jgvejqoj83jsw/new_data.conv?dl=1 > data/new_data.conv
    

fi

wait

if [[ $1 == "train" ]]; then
    if [[ $2 == "chinese" ]]; then
        python main.py train --chinese=True --attn=False
    elif [[ $2 == "english" ]]; then
        if [[ $3 == "normal" ]]; then
            python main.py train --chinese=False --fb=False --model_path="checkpoints/memory_new.pth"
        elif [[ $3 == "normal-low" ]]; then
            python main.py train --chinese=False --fb=False --model_path="checkpoints/memory.pth" --results_path="data/data.bin"
        elif [[ $3 == "attention" ]]; then
            python main.py train --attn=True --chinese=False --fb=False
        else
            python train_RL.py
        fi
    fi
elif [[ $1 == "test" ]]; then
    if [[ $2 == "chinese" ]]; then
        python main.py test --chinese=True --attn=False
    elif [[ $2 == "english" ]]; then
        if [[ $3 == "normal" ]]; then
            python main.py test --chinese=False --fb=False --model_path="checkpoints/memory_new.pth"
        elif [[ $3 == "normal-low" ]]; then
            python main.py test --chinese=False --fb=False --model_path="checkpoints/memory.pth" --results_path="data/data.bin"
        elif [[ $3 == "attention" ]]; then
            python main.py test --attn=True --chinese=False --fb=False
        else
            python main.py test --chinese=False --fb=False --rl=True
        fi
    fi
fi

