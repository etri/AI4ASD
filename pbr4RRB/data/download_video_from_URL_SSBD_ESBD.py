import csv
from pytube import YouTube
from pytube.exceptions import VideoPrivate, ExtractError, MembersOnly, PytubeError, VideoUnavailable

dataset = 'ESBD' #'SSBD

if dataset == 'SSBD':

    f = open('SSBD_refined.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    line_idx = 0
    for line in rdr:
        print(line)

        action_cls = line[1]
        file_name = line[3]
        file_tok = file_name.split("/")

        target_dir = '/home/ych/data/SS_Behaviours/Dataset_revised/' + action_cls
        target_file_name = target_dir + '/' + file_tok[-1] + '.mp4'

        try:
            yt = YouTube(file_name)
            stream = yt.streams.get_highest_resolution()
            stream.download(filename=target_file_name)
        except VideoPrivate as err:
            print(err)
            continue
        except VideoUnavailable as err:
            print(err)

    f.close()

elif dataset == 'ESBD':
    f = open('ESBD.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    line_idx = 0
    for line in rdr:
        print(line)

        file_tok = line[0].split('\t')

        action_cls = file_tok[1]
        file_name = file_tok[2]
        name_tok = file_name.split("/")

        target_dir = '/home/ych/data/ESBD/Dataset/' + action_cls
        target_file_name = target_dir + '/' + name_tok[-1] + '.mp4'

        try:
            yt = YouTube(file_name)
            stream = yt.streams.get_highest_resolution()
            stream.download(filename=target_file_name)
        except (Exception):
            print('video unavailable for some reasons')
            continue

    f.close()

