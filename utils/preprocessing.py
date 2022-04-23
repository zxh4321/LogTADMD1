# Drain: https://github.com/logpai/logparser

import os
import tarfile
import urllib.request
from . import Drain,DrainThunderbird


def parsing(dataset_name, output_dir='//Dataset'):
    """Download and parsing dataset

    Args:
        dataset_name: name of the log dataset
        output_dir: directory name for data storage

    Returns:
        Structured log data in Pandas Dataframe after adopt Drain
    """
    # 方法用于返回当前工作目录
    path = os.getcwd()
    # 构建日志解析后输出的目录
    directory = path + output_dir
    if not os.path.exists(directory):
        print('Making directory for dataset storage')
        os.makedirs(directory)

    if dataset_name == 'BGL':
        url = 'https://zenodo.org/record/3227177/files/BGL.tar.gz?download=1'
        downloaded_filename = 'BGL.tar.gz'
        print(downloaded_filename)
        if not os.path.isfile(downloaded_filename):
            urllib.request.urlretrieve(url, downloaded_filename)
        # tarfile模块用于解包和打包文件
        # r|gz 采用gzip格式解压并打开文件
        tar = tarfile.open(downloaded_filename, "r|gz")
        #TarFile.extractall(path=".",members = None,*,numeric_owner=false)
        #解包所有文件到当前目录，或path指定的目录，解包文件之前一定要确保安全，防止覆盖本地或上级目录等恶意行为
        tar.extractall()
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        input_dir = ''  # The input directory of log file
        output_dir = 'Dataset/'  # The output directory of parsing results
        log_file = 'BGL.log'  # The input log file name
        log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL log format
        # Regular expression list for optional preprocessing (default: [])
        regex = [
            r'core\.\d+',
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'([0-9a-f]+[:][0-9a-f]+)',
            r'fpr[0-9]+[=]0x[0-9a-f]+ [0-9a-f]+ [0-9a-f]+ [0-9a-f]+',
            r'r[0-9]+[=]0x[0-9a-f]+',
            r'[l|c|xe|ct]r=0x[0-9a-f]+',
            r'0x[0-9a-f]+',
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        st = 0.5  # Similarity threshold
        depth = 4  # Depth of all leaf nodes
        # 设置进行日志
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass

    elif dataset_name == 'Thunderbird':
        url = 'https://zenodo.org/record/3227177/files/Thunderbird.tar.gz?download=1'
        downloaded_filename = 'Thunderbird.tar.gz'
        if not os.path.isfile(downloaded_filename):
           urllib.request.urlretrieve(url, downloaded_filename)
        tar = tarfile.open(downloaded_filename, "r|gz")
        tar.extractall()
        tar.close()
        try:
            os.remove(downloaded_filename)
        except OSError:
            pass

        input_dir = ''  # The input directory of log file
        output_dir = 'Dataset/'  # The output directory of parsing results
        log_file = 'Thunderbird.log'  # The input log file name
        log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
        # Regular expression list for optional preprocessing (default: [])
        regex = [r'(\d+\.){3}\d+',
                 r'[a-d]n[0-9]+',
                 r'\<[0-9a-f]{16}\>\{.+\}']
        st = 0.3  # Similarity threshold
        depth = 2  # Depth of all leaf nodes

        parser = DrainThunderbird.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st,
                                            rex=regex)
        parser.parse(log_file)

        try:
            os.remove(log_file)
        except OSError:
            pass


