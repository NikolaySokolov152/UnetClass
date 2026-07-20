import numpy as np
from generator3D import get_cpu_numpy_from_list_of_tensor

def split_3d_data(data, shape, overlap):
    # data : CDHW
    c, d, h, w = data.shape
    # shape : dhw
    t_d, t_h, t_w = shape

    # overlap : value

    assert t_d <= d, "Z - dim error"
    assert t_h <= h, "Y - dim error"
    assert t_w <= w, "X - dim error"

    #prepare_stage

    def prepare_axis_start_points(temp_size, overlap):
        end_size = temp_size
        start_pos_list = []
        now_pos = 0
        while temp_size > 0:
            start_pos_list.append(now_pos)
            temp_size -= overlap
            now_pos += overlap
        start_pos_list.append(end_size)
        return start_pos_list
    
    start_z_pos_list = prepare_axis_start_points(d - t_d, overlap)
    start_y_pos_list = prepare_axis_start_points(h - t_h, overlap)
    start_x_pos_list = prepare_axis_start_points(w - t_w, overlap)

    end_z_pos_list = []
    end_y_pos_list = []
    end_x_pos_list = []

    frame_list = []
    for s_z in start_z_pos_list:
        e_z = s_z+t_d
        end_z_pos_list.append(e_z)
        for s_y in start_y_pos_list:
            e_y = s_y+t_h
            end_y_pos_list.append(e_y)
            for s_x in start_x_pos_list:
                e_x = s_x+t_w
                end_x_pos_list.append(e_x)
                frame_list.append(data[:, s_z:e_z, s_y:e_y, s_x:e_x])

    return frame_list, ((start_z_pos_list, start_y_pos_list, start_x_pos_list),
                        (end_z_pos_list, end_y_pos_list, end_x_pos_list))


def glue_data(result_list, split_data, shape_data):
    # result_list: list[CDHW], DHW = shape_data
    s_z, s_y, s_x = split_data[0]
    e_z, e_y, e_x = split_data[1]

    t_d, t_h, t_w = shape_data
    
    d = e_z[-1]
    h = e_y[-1]
    w = e_x[-1]
    c = result_list[0].shape[0]

    result_data = np.zeros((d, h, w, c), dtype = np.float32)

    def calc_overlaps(index, max_index_val, start_pos_list, end_pos_list):
        if index == 0:            # start
            left_overlap = 0
            right_overlap = (end_pos_list[index] - start_pos_list[index+1])//2
        elif index < max_index_val:  # mid
            left_overlap = (end_pos_list[index-1] - start_pos_list[index])//2
            right_overlap = (end_pos_list[index] - start_pos_list[index+1])//2
        else:                  # end
            left_overlap = (end_pos_list[index-1] - start_pos_list[index])//2
            right_overlap = 0
        return start_pos_list[index], end_pos_list[index], left_overlap, right_overlap   
    
    now_index = 0
    for k in range(len(s_z)):
        (start_overlap_z,
         end_overlap_z,
         left_overlap_z,
         right_overlap_z) = calc_overlaps(k, len(s_z)-1, s_z, e_z)

        for j in range(len(s_y)):
            (start_overlap_y,
             end_overlap_y,
             left_overlap_y,
             right_overlap_y) = calc_overlaps(j, len(s_y)-1, s_y, e_y)

            for i in range(len(s_x)):
               (start_overlap_x,
                end_overlap_x,
                left_overlap_x,
                right_overlap_x) = calc_overlaps(i, len(s_x)-1, s_x, e_x)

               data_from_model = result_list[now_index]
               data_on_host =  get_cpu_numpy_from_list_of_tensor(data_from_model)                
            
               result_data[start_overlap_z+left_overlap_z:end_overlap_z-right_overlap_z,
                            start_overlap_y+left_overlap_y:end_overlap_y-right_overlap_y,
                            start_overlap_x+left_overlap_x:end_overlap_x-right_overlap_x,
                            :] = data_on_host[left_overlap_z:t_d-right_overlap_z,
                                              left_overlap_y:t_h-right_overlap_y,
                                              left_overlap_x:t_w-right_overlap_x, :]
               now_index += 1

    return result_data # DHWC
    