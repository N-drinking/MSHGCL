# 模型-MSHGCL
以DGNN-DDI模型为基础，构建并添加了药物层间对比模块和药物层内对比模块
## Note
We have added comments to drugbank/data_preprocessing.py and drugbank/model.py. If you are interested in the technical details of preprocessing steps and algorithms, I think those comments would be helpful. 
## Requirements  pytorch版本的变动可能会导致模型性能的波动
numpy ==1.22.3          
pandas == 1.4.3           
python == 3.7.16             
pytorch == 1.11.0           
rdkit == 2022.03.3     
scikit-learn == 1.0.2                   
torch-geometric == 2.2.0                  
torch-scatter == 2.0.9                  
torch-sparse == 0.6.13            
tqdm == 4.66.1
## 使用说明  
- 第一步数据预处理:
- 在terminal终端窗口使用 ' python data_preprocessing.py -d drugbank -o all`命令传递参数来运行 data_preprocessing.py 文件完成数据预处理
- 如果想要在小数据集上运行测试，在 data_preprocessing.py 和 dataset.py 文件中将路径 'data/drugbank.tab' 修改为 'data/Deng's dataset/dataset_small.tab' 并使用终端重新运行 data_preprocessing.py 文件。

- 第二步训练:
- 运行 train.py 文件进行训练，训练结果保存在 'save/logs/train.log'
- 如果想要新建记录文件，可以通过修改 log/train_logger.py 文件中的 self.log_path属性来自定义文件名，修改完成后在运行 train.py 文件时会自动创建新的记录文件。

