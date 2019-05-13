# Download Data

if [ -d "data/" ]; then
    echo "The data had been already downloaded"
else
    mkdir data/
    
    curl -L https://www.dropbox.com/sh/ffb8ohnb0j7cb15/AACDKNel7WVK0nGo5M6lm1Jfa?dl=1 > download1.zip
    unzip download1.zip
    rm download1.zip

    curl -L https://www.dropbox.com/sh/ffq6vkdnx8rmfj6/AABJ7JciSij-_VVksYuoRcAca?dl=1 > data/download2.zip
    unzip data/download2.zip -d data
    rm data/download2.zip

fi