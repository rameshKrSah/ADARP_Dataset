import os
import numpy as np
import csv
from argparse import ArgumentParser

# local imports
import utils as utl
import filters as filters
import preprocessing as preprocessing

participants_folder_names = ['Part 101C',
 'Part 102C',
 'Part 104C',
 'Part 105C',
 'Part 106C',
 'Part 107C',
 'Part 108C',
 'Part 109C',
 'Part 110C',
 'Part 111C',
 'Part 112C']


# # Details
# The sampling rate for
# - Galvanic Skin Response is 4hz
# - Skin Temperature is 4hz
# - Blood Volume Pulse is 64hz
# - Acceleration is 32hz
# - Heart rate is 1hz
#
# For each sensor csv file, the first row is the timestamp for recording start time
# and in some sensor files the second row is sampling frequency. We can use the
# timestamp to put time values next to each row of values in all sensor files,
# and use this timestamp to extract the window around the tag timestamps.
#
# For window size we can experiment with different values, and we will start
# with 25 seconds window and go upto 10 minutes.
#
# We also need to apply filtering on sensor values. For EDA values,
# 1. First-order BW LPF cut-off frequency 5 Hz to remove noise.
# 2. First-order BW HPF cut-off frequency 0.05 Hz to separate SCR and SCL
#
# and for skin temperature
# 1. Second-order BW LPF frequency of 1 Hz
# 2. Second-order BW HPF frequency of 0.1 Hz
#
#
# Unix time is a system for describing a point in time, and is the number of
# seconds that have elapsed since the Unix epoch, minus leap seconds; the Unix
# epoch is 00:00:00 UTC on 1 January 1970.
#
# Every file except the IBI file has sampling frequency in the second row.
# All files have staring time in UNIX timestamp in the first row.

# Constants
E4_EDA_SF = 4
E4_ACC_SF = 32
E4_BVP_SF = 64
E4_HR_SF = 1
E4_TEMP_SF = 4

EDA_CUTOFF_FREQ = 5.0/ E4_EDA_SF

# 40 minutes, 20 minutes before the event and 20 minutes after the event
tag_segment_length_seconds = 40 * 60

# overlapping window segmentation details 
window_length_seconds = 60
overlap_seconds = 30          # 50% overlap
overlap_percent = 0.5

# not-stress data is extracted 60 minutes before and after the event markers.
not_stress_buffer_from_tag = 60 * 60

# data_folder = "../Data/Wearable Devices Study Data/"
# output_folder = "../Processed Data/24 seconds window ADARP/"

def get_sensor_data(file_path):
    """
    Load data from a text file located at file_path.
    :param file_path: path to the text file

    """
    data = []
    try:
        data = np.genfromtxt(file_path, delimiter=',')
    except:
        print("Error reading the file {}".format(file_path))

    return data


def get_tag_timestamps(tag_file):
    """
        Open the tag files and retun the tag timestamps as an array.
    
    :param tag_file: Path to the tags file.
    """

    tag_timestamps = []

    # count = 0
    # for line in open(tag_file): count += 1

    # if count < 2:
    #     return tag_timestamps

    # print(f"{count - 1} tags in {tag_file}")
    with open(tag_file, "r") as read_file:
        csv_reader = csv.reader(read_file)
        # skip the header line
        # next(csv_reader)
        for row in csv_reader:
            # print(row)
            unix_time = float(row[0])
            tag_timestamps.append(unix_time)

    return tag_timestamps


def verify_tags(data_folder):
    total_timestamps = 0
    participants_tags = {}

    for p in participants_folder_names:
        participants_folder_path = data_folder + p + "/"
        part_subfolders = os.listdir(participants_folder_path)
        # print(p)

        temp = 0
        # for each sub-folder in the participants folder
        for sub in part_subfolders:
            path = participants_folder_path + sub

            # get the tag events in this folder
            tag_timestamps = get_tag_timestamps(path + "/tags.csv")

            temp += len(tag_timestamps)
            total_timestamps += len(tag_timestamps)

        participants_tags[p] = temp
        # break
    return participants_tags, total_timestamps


def extract_segments_around_tags(data, tags, segment_size):
    """
        Given data array, tags array and window size extract window size segments 
        from the data array around the tags.

    :param data: Data array
    :param tags: An array with tag event times
    :param segment_size: Segment size in seconds

    """
    # return array
    segments = []
    
    # get the start time: expressed as unit timestamp in UTC i.e., seconds from Jan 1 1970
    start_time = data[0]
    
    # get the sampling frequency expressed in Hz
    sampling_freq = data[1]
    
    try:
        if len(start_time):
            start_time = start_time[0]
    except:
        start_time = start_time
    
    try:
        if len(sampling_freq):
            sampling_freq = sampling_freq[0]
    except:
        sampling_freq = sampling_freq

    # get the sensor data and data length
    sensor_data = data[2:]
    data_length = len(sensor_data)

    # the timestamp corresponding to the last data value
    end_time = start_time + (data_length / sampling_freq)

    # the number of data samples before and after the timestamps
    n_obs = int((segment_size // 2) * sampling_freq)

    skipped_tags = 0

    # for each time stamp in tags
    for timestamp in tags:
        # if the timestamp is within the sensor time array
        if (timestamp >= start_time) & (timestamp <= end_time):
            # how far is the timestamp from the start time.
            difference = int(timestamp - start_time)

            # get the index in the sensor data array, based on the difference of tag timestamp
            position = int(difference * sampling_freq)
            
            # check is there are enough data points to extract 1 hour of data around the tag. 
            check_threshold = 30 * 60 *  sampling_freq        # 30 minutes before and after the tag
            if ((position - check_threshold) < 0) | ((position + check_threshold) > data_length):
                # print("not enough data for 1 hour segment length")
                skipped_tags += 1
                continue

            # window segment position in the data array
            from_ = position - n_obs
            to_ = position + n_obs

            # here we can insert logic to make sure that we have sensor data of length segment_size
            if (from_ < 0) | (to_ > data_length):
                # skip this segment if length is not equal to the segment_size
                print(f"skipping segment From: {from_}, To: {to_}, Data Len: {data_length}")
                continue
            else:
                # get the data segment
                seg = sensor_data[from_:to_]
                segments.append(seg)

            # necessary to make sure that we dont have invalid indices
            # if (from_ < 0):
            #     from_ = 0
            # if (to_ > data_length):
            #     to_ = data_length

    # if skipped_tags != 0:
    #     print(f"Skipped {skipped_tags} ", end=" ")
    return segments


def get_eda_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get EDA segments from the EDA CSV file in data_folder with tag_timestamps 
        for segment length of segment_size

    :param data_folder: Path to the folder containing the EDA file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment size in seconds

    """

    # load the data from EDA.csv
    file_path = data_folder + "/EDA.csv"
    processed_EDA = []

    sensor_data = get_sensor_data(file_path)

    if len(sensor_data) == 0:
        return processed_EDA

    segments = extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)
    for p in segments:
        pp = filters.butter_lowpassfilter(np.array(p).ravel(), EDA_CUTOFF_FREQ, E4_EDA_SF, order=2)
        pp = preprocessing.normalization(pp)
        processed_EDA.append(pp)

    return processed_EDA


def get_hr_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get HR segments from the HR CSV file in data_folder with tag_timestamps 
        for segment length of segment_size

    :param data_folder: Path to the folder containing the HR file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Window size in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/HR.csv"
    sensor_data = get_sensor_data(file_path)
    if len(sensor_data) == 0:
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)


def get_temp_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get TEMP segments from the TEMP CSV file in data_folder with tag_timestamps 
        for segment length of segment_size

    :param data_folder: Path to the folder containing the TEMP file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment length in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/TEMP.csv"
    sensor_data = get_sensor_data(file_path)
    
    if len(sensor_data) == 0:
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)


def get_bvp_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get BVP segments from the BVP CSV file in data_folder with tag_timestamps 
        for segment length of segment_size

    :param data_folder: Path to the folder containing the BVP file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment length in seconds.

    """

    # load the data from EDA.csv
    file_path = data_folder + "/BVP.csv"
    sensor_data = get_sensor_data(file_path)

    if len(sensor_data) == 0:
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)


def get_acc_data_around_tags(data_folder, tag_timestamps, segment_size):
    """
        Get ACC segments from the ACC CSV file in data_folder with tag_timestamps 
        for segment length of segment_size

    :param data_folder: Path to the folder containing the ACC file
    :param tag_timestamps: An array containing the tag event markers.
    :param segment_size: Segment length in seconds.

    """
    # load the data from EDA.csv
    file_path = data_folder + "/ACC.csv"
    sensor_data = get_sensor_data(file_path)
    
    if len(sensor_data) == 0:
        return []
    else:
        return extract_segments_around_tags(sensor_data, tag_timestamps, segment_size)


def extract_segments_for_verified_tags(data_folder, tag_timestamps_folder, segment_length, output_folder=None):
    """
        Extract sensor segment around tag event markers. 

        Params
        data_folder -- path to the complete dataset
        tag_timestamps_folder -- path to the folder containing the tag event markers that are verified.
        segment_length -- length of the sensor segment in seconds
        output_folder -- path to store the extracted sensor segment. default None (do not save)

        Return
        Sensor segments for EDA, BVP, ACC, HR, and TEMP
    """
    
    # total number of segments
    total_segments = 0

    # data containers
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data = []

    for participants_tags_file in os.listdir(tag_timestamps_folder):
        # get the verified tag for the participants
        tag_events = get_tag_timestamps(tag_timestamps_folder  + participants_tags_file)
        if(len(tag_events) == 0):
            continue

        # get the participants identifier
        participant_name = participants_tags_file[:9]
        #print(f"{participant_name} has verified tags {tag_events}")

        # the original folder with participant data
        participants_data_folder = data_folder + participant_name + "/"

        # subfolders within the participants data folder. 
        subfolders = os.listdir(participants_data_folder)

        # for each verified tag search all the participants subfolders for matching event markers.
        for tag in tag_events:
            #print(f"Searching for verified tag {tag}")

            # for each sub-folder in the participants folder
            for sub in subfolders:
                sub_folder_path = participants_data_folder + sub

                # get the tag events in this folder
                tag_timestamps = get_tag_timestamps(sub_folder_path + '/tags.csv')
                if(len(tag_timestamps) == 0):
                    continue

                # print(f"{participant_name} event markers {tag_timestamps}")
                # if there are tag events, and if any verified tags are within this list
                # extract data around the verified tag event timestamp
                for stamps in tag_timestamps:
                    if abs(tag - stamps) < 5:
                        #print(f"Verified tag {tag}, event marker {stamps}")
                        total_segments += 1

                        # first EDA
                        values = get_eda_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            eda_data.extend(values)

                        # second temperature
                        values = get_temp_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            temp_data.extend(values)

                        # third bvp
                        values = get_bvp_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            bvp_data.extend(values)

                        # fourth hr
                        values = get_hr_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            hr_data.extend(values)

                        # fifth acc
                        values = get_acc_data_around_tags(sub_folder_path, [stamps], segment_length)
                        if len(values):
                            acc_data.extend(values)

        # save the participants data
        # if output_folder != None:
        #     print("Saving data of participants " + p)
        #     utl.save_data(output_folder + p + "_EDA_TAG.pkl", np.array(part_eda_data))
        #     utl.save_data(output_folder + p + "_TEMP_TAG.pkl", np.array(part_temp_data))
        #     utl.save_data(output_folder + p + "_HR_TAG.pkl", np.array(part_hr_data))
        #     utl.save_data(output_folder + p + "_BVP_TAG.pkl", np.array(part_bvp_data))
        #     utl.save_data(output_folder + p + "_ACC_TAG.pkl", np.array(part_acc_data))

    return np.array(eda_data), np.array(hr_data), np.array(acc_data), np.array(bvp_data), np.array(temp_data)


def extract_data_around_tags(data_folder, segment_length, save_part_data=False, output_folder=None, 
                            include_eda=False, include_bvp=False, include_acc=False, include_hr=False,
                            include_temp=False):
    """
        Extract sensor segment around tag event markers.

        Param
        ===================
        data_folder -- path to the data
        segment_length -- length of the sensor segment to extract in seconds
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)

        Return
        ===================
        Sensor segment for EDA, BVP, HR, ACC, and TEMP

    """
    # data containers
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data= []

    # for each participants
    for p in participants_folder_names:
        # participants data container
        part_eda_data = []
        part_hr_data = []
        part_acc_data = []
        part_bvp_data = []
        part_temp_data = []

        #print("Extracting data for participants: {}".format(p))
        participants_folder_path = data_folder + p + "/"
        part_subfolders = os.listdir(participants_folder_path)

        # for each sub-folder in the participants folder
        for sub in part_subfolders:
            path = participants_folder_path + sub
#             #print("For subfolder: {}".format(path))

            # get the tag events in this folder
            tag_timestamps = get_tag_timestamps(path + "/tags.csv")

            # if there are tag events, get the sensor values
            if len(tag_timestamps):
                if include_eda:
                    # first EDA
                    values = get_eda_data_around_tags(path, tag_timestamps, segment_length)
                    if len(values):
                        eda_data.extend(values)
                        part_eda_data.extend(values)

                if include_temp:
                    # second temperature
                    values = get_temp_data_around_tags(path, tag_timestamps, segment_length)
                    if len(values):
                        temp_data.extend(values)
                        part_temp_data.extend(values)

                if include_bvp:
                    # third bvp
                    values = get_bvp_data_around_tags(path, tag_timestamps, segment_length)
                    if len(values):
                        bvp_data.extend(values)
                        part_bvp_data.extend(values)

                if include_hr:
                    # fourth hr
                    values = get_hr_data_around_tags(path, tag_timestamps, segment_length)
                    if len(values):
                        hr_data.extend(values)
                        part_hr_data.extend(values)

                if include_acc:
                    # fifth acc
                    values = get_acc_data_around_tags(path, tag_timestamps, segment_length)
                    if len(values):
                        acc_data.extend(values)
                        part_acc_data.extend(values)

        #print(f"Processed participants {p} # eda segments {len(part_eda_data)}")
        #print("Total # eda segments ", len(eda_data))

        # save the participants data
        if (save_part_data)  & (output_folder != None):
            #print("Saving data of participants " + p)
            utl.save_data(output_folder + p + "_EDA_TAG.pkl", np.array(part_eda_data))
            utl.save_data(output_folder + p + "_TEMP_TAG.pkl", np.array(part_temp_data))
            utl.save_data(output_folder + p + "_HR_TAG.pkl", np.array(part_hr_data))
            utl.save_data(output_folder + p + "_BVP_TAG.pkl", np.array(part_bvp_data))
            utl.save_data(output_folder + p + "_ACC_TAG.pkl", np.array(part_acc_data))

    if include_eda & include_temp & include_hr & include_bvp & include_acc:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data), np.array(acc_data)

    elif include_eda & include_temp & include_hr & include_bvp:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data)

    elif include_eda & include_temp & include_hr:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data)

    elif include_eda & include_temp:
        return np.array(eda_data), np.array(temp_data)

    elif include_eda:
        return np.array(eda_data)


def get_segments_between_timestamps(data_array, tag_timestamps, pre_and_post_event_marker_len=60*60, segments=[]):
    """
        Extract sensor segment for the not-stress class between event markers. For a given event marker 
        timestamp we extract sensor segment until one hour before the event marker and one hour after the event
        marker.

        Param
        ================================
        data_array -- sensor data array
        tag_timestamps -- timestamps of tags to extract data around of
        pre_and_post_event_marker_len -- Time duration to skip data points pre and post event marker
        segments -- Array to store the extracted segments
    """

    if(len(data_array) == 0):
        return segments
    
    if len(tag_timestamps) == 0:
        segments.append(data_array[2:])
        return segments
    else:
        # extract start time, sampling freq, and n_observations
        start_time = data_array[0]
        sampling_freq = data_array[1]
        try:
            if len(start_time):
                start_time = start_time[0]
        except:
            start_time = start_time

        try:
            if len(sampling_freq):
                sampling_freq = sampling_freq[0]
        except:
            sampling_freq = sampling_freq

        # number of samples to skip before and after the event
        n_observation = int(pre_and_post_event_marker_len * sampling_freq)
        
        # create the tags, add the start and end time into tags
        tags = [start_time]
        tags.extend(tag_timestamps)
        tags.append(tags[0] + len(data_array) / sampling_freq)
        
        # sensor data and the length
        data = data_array[2:]
        data_length = len(data)

        # for each tag in the tags array
        for i in range(len(tags)):
            j = i + 1
            if j >= len(tags):
                # if at the end, break free
                break
            
            # get the starting and end point for the sensor segment.
            start_tag = tags[i] # this is the position of start tag
            end_tag = tags[j] # this is the position of the end tag

#             print("Current tags pair ", (start_tag, end_tag))
            # the positions in the array
            here_ = int((start_tag - start_time) * sampling_freq + n_observation)  # pre_and_post_event_marker_len after the event
            there_ = int((end_tag - start_time) * sampling_freq - n_observation) # pre_and_post_event_marker_len before the event

#             print("Indices ", (here_, there_))
            # if there are data points between the start and end points, extract those data points else ignore them
            if((there_ - here_) > 0):
                pp = data[here_:there_]
                segments.append(pp)

        return segments


def not_stressed_data_from_all_files(data_folder, segment_length_to_skip, save_part_data=False, output_folder=None, segment=False, 
                                           include_eda=False, include_bvp=False, include_acc=False, include_hr=False,
                                           include_temp=False):
    """
        Extract data for not-stressed class from all folders.

        Param
        ===================
        data_folder -- path to the data
        segment_length_to_skip -- length of time to skip before and after an tag event
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)
        segment -- whether to run sliding window or not

        Return
        ===================
        Sensor segment for EDA, BVP, HR, ACC, and TEMP
    """

    # data containers
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data= []

    # for each participants
    for p in participants_folder_names:
        # participants data container
        part_eda_data = []
        part_hr_data = []
        part_acc_data = []
        part_bvp_data = []
        part_temp_data = []

        #print("Extracting data for participants: {}".format(p))
        participants_folder_path = data_folder + p + "/"
        part_subfolders = os.listdir(participants_folder_path)

        # for each sub-folder in the participants folder
        for sub in part_subfolders:
            path = participants_folder_path + sub
#             print("For subfolder: {}".format(path))

            # get the tag events in this folder
            tag_timestamps = get_tag_timestamps(path + "/tags.csv")

            if include_eda:
                # load the EDA data
                data = get_sensor_data(path+"/EDA.csv")
                if len(data) != 0:
                    part_eda_data = get_segments_between_timestamps(data, tag_timestamps, segment_length_to_skip, part_eda_data)

            if include_hr:
                # HR Segments
                data = get_sensor_data(path+"/HR.csv")
                if len(data) != 0:
                    part_hr_data = get_segments_between_timestamps(data, tag_timestamps, segment_length_to_skip, part_hr_data)

            if include_temp:
                # TEMP Segments
                data = get_sensor_data(path+"/TEMP.csv")
                if len(data) != 0:
                    part_temp_data = get_segments_between_timestamps(data, tag_timestamps,segment_length_to_skip, part_temp_data)

            if include_bvp:
                # BVP Segments
                data = get_sensor_data(path+"/BVP.csv")
                if len(data) != 0:
                    part_bvp_data = get_segments_between_timestamps(data, tag_timestamps, segment_length_to_skip, part_bvp_data)

            if include_acc:
                # ACC Segments
                data = get_sensor_data(path+"/ACC.csv")
                if len(data) != 0:
                    part_acc_data = get_segments_between_timestamps(data, tag_timestamps, segment_length_to_skip, part_acc_data)
        
        # We filter and normalize the EDA data. 
        processed_EDA =[]
        for dt in part_eda_data:
            pp = filters.butter_lowpassfilter(np.array(dt).ravel(), EDA_CUTOFF_FREQ, E4_EDA_SF, order=2)
            pp = preprocessing.normalization(pp)
            processed_EDA.append(pp)
        
        if (save_part_data)  & (output_folder != None):
            # save the participants data
            #print("Saving data of participants " + p)                                          
            utl.save_data(output_folder + p + "_EDA_NO_TAG.pkl", np.array(processed_EDA))
            utl.save_data(output_folder + p + "_TEMP_NO_TAG.pkl", np.array(part_temp_data))
            utl.save_data(output_folder + p + "_HR_NO_TAG.pkl", np.array(part_hr_data))
            utl.save_data(output_folder + p + "_BVP_NO_TAG.pkl", np.array(part_bvp_data))
            utl.save_data(output_folder + p + "_ACC_NO_TAG.pkl", np.array(part_acc_data))
    
        # add the participants data to the whole data
        eda_data.extend(processed_EDA)
        hr_data.extend(part_hr_data)
        temp_data.extend(part_temp_data)
        bvp_data.extend(part_bvp_data)
        acc_data.extend(part_acc_data)
        
        #print(f"Processed participants {p} data {len(part_eda_data)}")
        #print("Total data ", len(eda_data))

    if include_eda & include_temp & include_hr & include_bvp & include_acc:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data), np.array(acc_data)

    elif include_eda & include_temp & include_hr & include_bvp:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data)

    elif include_eda & include_temp & include_hr:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data)

    elif include_eda & include_temp:
        return np.array(eda_data), np.array(temp_data)
        
    elif include_eda:
        return np.array(eda_data)

def not_stressed_data_from_zero_tags_files(data_folder, save_part_data=False, output_folder=None, segment=False, 
                                           include_eda=False, include_bvp=False, include_acc=False, include_hr=False,
                                           include_temp=False):
    """
        Extract data for not-stressed class from files with zero tag events
        Param
        ===================
        data_folder -- path to the data
        save_part_data -- whether to save the participants data or not (default - false)
        output_folder -- path to the directory to save the data (default - none)
        segment -- whether to run sliding window or not

        Return
        ===================
        Sensor segment for EDA, BVP, HR, ACC, and TEMP
    """
    # data containers
    eda_data = []
    hr_data = []
    acc_data = []
    bvp_data = []
    temp_data= []
    
    # for each participants
    for p in participants_folder_names:
        part_eda_data = []
        part_hr_data = []
        part_acc_data = []
        part_bvp_data = []
        part_temp_data = []
        
        #print("Extracting data for participants {}".format(p))
        participants_folder_path = data_folder + p + "/"
        subfolders = os.listdir(participants_folder_path)
        
        # for each subfolders in the participant folder
        for sub in subfolders:
            path = participants_folder_path + sub
            # get tag timestamps
            tag_timestamps = get_tag_timestamps(path + "/tags.csv")
            
            if len(tag_timestamps) == 0:
                if include_eda:
                    # load the EDA data
                    data = get_sensor_data(path+"/EDA.csv")
                    if len(data) != 0:
                        part_eda_data.append(data[2:])

                if include_hr:
                    # HR Segments
                    data = get_sensor_data(path+"/HR.csv")
                    if len(data) != 0:
                        part_hr_data.append(data[2:])
                
                if include_temp:
                    # TEMP Segments
                    data = get_sensor_data(path+"/TEMP.csv")
                    if len(data) != 0:
                        part_temp_data.append(data[2:])
                
                if include_bvp:
                    # BVP Segments
                    data = get_sensor_data(path+"/BVP.csv")
                    if len(data) != 0:
                        part_bvp_data.append(data[2:])
                
                if include_acc:
                    # ACC Segments
                    data = get_sensor_data(path+"/ACC.csv")
                    if len(data) != 0:
                        part_acc_data.append(data[2:])
                
        # We filter and normalize the EDA data. 
        processed_EDA =[]
        for p in part_eda_data:
            pp = filters.butter_lowpassfilter(np.array(p).ravel(), EDA_CUTOFF_FREQ, E4_EDA_SF, order=2)
            pp = preprocessing.normalization(pp)
            processed_EDA.append(pp)
        
        if (save_part_data)  & (output_folder != None):
            # save the participants data
            #print("Saving data of participants " + p)                                          
            utl.save_data(output_folder + p + "_EDA_NO_TAG.pkl", np.array(processed_EDA))
            utl.save_data(output_folder + p + "_TEMP_NO_TAG.pkl", np.array(part_temp_data))
            utl.save_data(output_folder + p + "_HR_NO_TAG.pkl", np.array(part_hr_data))
            utl.save_data(output_folder + p + "_BVP_NO_TAG.pkl", np.array(part_bvp_data))
            utl.save_data(output_folder + p + "_ACC_NO_TAG.pkl", np.array(part_acc_data))
        
        # add the participants data to the whole data
        eda_data.extend(processed_EDA)
        hr_data.extend(part_hr_data)
        temp_data.extend(part_temp_data)
        bvp_data.extend(part_bvp_data)
        acc_data.extend(part_acc_data)
        
        #print(f"Processed participants {p} data {len(processed_EDA)}")
        #print("Total data ", len(eda_data))

    if include_eda & include_temp & include_hr & include_bvp & include_acc:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data), np.array(acc_data)

    elif include_eda & include_temp & include_hr & include_bvp:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data), np.array(bvp_data)

    elif include_eda & include_temp & include_hr:
        return np.array(eda_data), np.array(temp_data), np.array(hr_data)

    elif include_eda & include_temp:
        return np.array(eda_data), np.array(temp_data)
        
    elif include_eda:
        return np.array(eda_data)


def segment_sensor_data(data_array, sample_rate, window_duration, overlap_percent):
    """
        Overlapping segmentation of data.
    @param data_array: Data to be segmented
    @param sample_rate: Sampling frequency
    @param window_duration: Window size in seconds
    @param overlap_percent: Overlap percentage between consequtive windows.
    """

    window_segments = np.zeros((1, sample_rate * window_duration))
    
    # get the window segments
    for dt in data_array:
#         print("Current data length ", len(dt))
        segments = utl.segment_sensor_reading(dt, window_duration, overlap_percent, sample_rate)
        if(len(segments)):
            window_segments = np.concatenate([window_segments, segments])
    
    # return the segments
    window_segments = window_segments[1:, ]
    return window_segments

def segment_sensor_data(data_array, sample_rate, window_duration, overlap_percent, samples_to_drop=None):
    """
        Overlapping segmentation of data with provision to drop samples at defined indices.
    @param data_array: Data to be segmented
    @param sample_rate: Sampling frequency
    @param window_duration: Window size in seconds
    @param overlap_percent: Overlap percentage between consequtive windows.
    @param samples_to_drop: indices of sample that needs to be dropped
    """

    window_segments = np.zeros((1, sample_rate * window_duration))
    dropped_segments = np.zeros((1, sample_rate * window_duration))
    
    # get the window segments
    for dt in data_array:
#         print("Current data length ", len(dt))
        segments = utl.segment_sensor_reading(dt, window_duration, overlap_percent, sample_rate)
        
        # drop any samples
        if samples_to_drop != None:
            dropped_segments = np.concatenate((dropped_segments, segments[samples_to_drop].reshape(1, sample_rate * window_duration)))

            try:
                segments = np.delete(segments, samples_to_drop, axis=0)
            except:
                # sometimes we at the end of very begining of the sensor data and there are not enough data stress segment_length > 60
                print(f"Stress windows shape {segments.shape}")
                segments = []

        if(len(segments)):
            window_segments = np.concatenate([window_segments, segments])
    
    # return the segments
    window_segments = window_segments[1:, ]
    return window_segments, dropped_segments[1:, ]

if __name__ == '__main__':
    parser = ArgumentParser("Data processing ADARP")
    parser.add_argument(
        "-i",
        "--input_directory",
        type=str,
        required=True,
        help="Directory that contains the subject data for ADARP"
    )

    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        required=True,
        help="Directory to store the processed data"
    )

    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        required=True,
        help="window size in seconds"
    )

    args = parser.parse_args()
    # args.input_directory
    # args.output_directory
    # args.window_size