#!/bin/bash

installCUDA() {
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
  fi
}

installPythonPackages() {
  if [[ -z $GPU ]]; then
    conda install pytorch torchvision cpuonly ignite tqdm -c pytorch
  else
    conda install pytorch torchvision ignite tqdm -c pytorch
  fi
}

makeDirectories() {
  mkdir -p checkpoints dataset
}

getDataset() {
  cd dataset
  curl -OL haochen.lu/drive/dataLM.zip
  unzip dataLM.zip
  cd ..
}

installCUDA
installPythonPackages
makeDirectories
getDatases
