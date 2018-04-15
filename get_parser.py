import argparse
#  "解析控制台参数"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default='/home/wangbo/bp/features_data/',help="原始文件存在的位置")
parser.add_argument('--train_test1_path',type=str,default='/home/wangbo/bp/',help="模型训练主要数据集存在的位置")
parser.add_argument('--train_rate1',type=float,default=0.2,help='模型训练主要数据集中训练集所占的比例')
parser.add_argument('--train_rate2',type=float,default=0.3,help='模型评估数据集中训练集所占的比例')
parser.add_argument('--data_update',type=bool,default=False,help='是否更新训练数据集？')
parser.add_argument('--use_fp16',type=bool,default=False,help='是否使用float16格式?')
parser.add_argument('--num_preprocess_threads',type=int,default=1,help='是否使用float16格式?')
parser.add_argument('--model',type=str,default="train",help='对模型是训练还是测试')
parser.add_argument('--bptype',type=str,default="dia",help='对模型是训练还是测试')

FLAGS=parser.parse_args()