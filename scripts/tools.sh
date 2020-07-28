#!/bin/bash

installCUDA() {
	source /etc/os-release
	if [[ $GPU == "nvidia" ]]; then
		echo "Installing Nvidia-Docker and CUDA toolkit"
		case $ID in
		arch)
			sudo pacman -Sy --needed --noconfirm cuda
			;;
		ubuntu | debian)
			wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
			sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
			sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
			sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

			sudo apt-get update
			sudo apt-get -y install cuda
			;;
		esac

		echo '=> Installing Nvidia Apex'
		CWD=$PWD

		cd /tmp
		git clone https://github.com/NVIDIA/apex
		cd apex
		pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
		cd ..
		rm -rf apex

		cd $CWD
	fi
}

installConda() {
	curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
	bash miniconda.sh -b -p $HOME/.conda
	rm -f miniconda.sh
	echo 'eval "$($HOME/.miniconda/bin/conda shell.zsh hook)"' >>~/.zshrc
	eval "$($HOME/.conda/bin/conda shell.zsh hook)"
}

installPythonPackages() {
	conda env create --name zerolm --file=environment.yml
	if [[ -z $GPU ]]; then
		conda install pytorch torchvision cpuonly -c pytorch
	else
		conda install pytorch torchvision -c pytorch
	fi
}

makeDirectories() {
	mkdir -p checkpoints dataset logs
}

getDataset() {
	cd dataset
	curl -OL haochen.lu/drive/dataLM.zip
	unzip dataLM.zip
	cd ..
}
