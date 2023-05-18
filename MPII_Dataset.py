import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skimage.transform
import copy


class MPII_Dataset(torch.utils.data.Dataset):

    def __init__(self, path="./MPII_dataset", split="train"):
        self.path = path
        self.split = split

        # Load data from the mat file
        self.mat_data = scipy.io.loadmat(os.path.join(path, "mpii_human_pose_v1_u12_1.mat"))

        # Load and store images (float) into a list
        self.dataset_size = 2000    # total 7131 Single man image in 24986 image   # self.dataset_size*0.2 need to be int
        self.max_h, self.max_w = 196, 196
        self.array_of_images = np.empty([self.dataset_size, self.max_h, self.max_w, 3], dtype=float)
        self.array_of_labels = np.empty([self.dataset_size, 3, 14], dtype=float)  # N x (X,Y,is_visible) x (14 joints)

        counter = 0
        for i, anno in enumerate(self.mat_data['RELEASE']['annolist'][0, 0][0]):
            img_fn = anno['image']['name'][0, 0][0]
            if 'annopoints' in str(anno['annorect'].dtype):
                annopoints = anno['annorect']['annopoints'][0]
                if annopoints.size == 1:
                    # Make sure there is only one person in each image
                    # Get the position and visual information of the 16 joint points
                    anno_point = annopoints[0]['point'][0, 0]
                    joint_pos_vis = np.zeros((3, 16))
                    j_id = [j_i[0, 0] for j_i in anno_point['id'][0]]
                    x = [x[0, 0] for x in anno_point['x'][0]]
                    y = [y[0, 0] for y in anno_point['y'][0]]

                    if 'is_visible' in str(anno_point.dtype):
                        vis = [v[0] if v.size > 0 else [0] for v in anno_point['is_visible'][0]]
                    else:
                        vis = [[0] for x in range(16)]

                    for _j_id, _x, _y, _vis in zip(j_id, x, y, vis):
                        joint_pos_vis[0][_j_id] = float(_x)
                        joint_pos_vis[1][_j_id] = float(_y)
                        joint_pos_vis[2][_j_id] = float(_vis[0])

                    # Reducing 16 joints to 14
                    joint = np.delete(joint_pos_vis, [6, 7], axis=1)
                    # Determining whether a label contains 14 joints
                    if np.any(joint[:2, :] == 0.0):
                        continue

                    # Image cropping
                    Original_img = plt.imread(os.path.join(path, "mpii_human_pose_v1_images", img_fn))
                    objpos = anno['annorect']['objpos'][0]
                    objpos_x = objpos[0][0, 0]['x'][0][0]
                    objpos_y = objpos[0][0, 0]['y'][0][0]
                    img_h = Original_img.shape[0]
                    img_w = Original_img.shape[1]
                    min_h = min(img_h - objpos_y, objpos_y)
                    min_w = min(img_w - objpos_x, objpos_x)
                    
                    x_left = int(objpos_x) - int(min_w)
                    x_right = int(objpos_x) + int(min_w)
                    y_lower = int(objpos_y) - int(min_h)
                    y_upper = int(objpos_y) + int(min_h)

                    Image_cropped = Original_img[y_lower:y_upper, x_left:x_right, :]
                    joint_cropped = copy.deepcopy(joint)
                    joint_cropped[0, :] = joint_cropped[0, :] - x_left
                    joint_cropped[1, :] = joint_cropped[1, :] - y_lower

                    # Image scale_and_pad
                    img, labels = self.scale_and_pad(Image_cropped, joint_cropped[:2, :])
                    self.array_of_images[counter] = img
                    self.array_of_labels[counter, :2, :] = labels
                    self.array_of_labels[counter, 2, :] = joint_cropped[2, :]
                    counter += 1
                    '''
                    # plot the image
                    img_Original = plt.imread(os.path.join(path, "mpii_human_pose_v1_images", img_fn))
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.title.set_text('Original_%s' % img_fn)
                    ax1.imshow(img_Original)
                    for t in range(14):
                        if joint[2, t] == 0.0:
                            c = 'b'
                        else:
                            c = 'r'
                        ax1.plot(joint[0, t], joint[1, t], '.', color=c)
                    ax1.plot(objpos_x, objpos_y, '.', color='g')

                    ax2.title.set_text('Image_cropped')
                    ax2.imshow(Image_cropped)
                    for t in range(14):
                        if joint_cropped[2, t] == 0.0:
                            c = 'b'
                        else:
                            c = 'r'
                        ax2.plot(joint_cropped[0, t], joint_cropped[1, t], '.', color=c)

                    ax3.title.set_text('Normalized_%s' % img_fn)
                    ax3.imshow(img)
                    for t in range(14):
                        if joint_cropped[2, t] == 0.0:
                            c = 'b'
                        else:
                            c = 'r'
                        ax3.plot(self.max_h * (0.5 + labels[0, t]), self.max_w * (0.5 + labels[1, t]), '.', color=c)

                    fig.tight_layout()
                    plt.show()
                    '''
                    print(counter, "/", self.dataset_size)
                    if counter == self.dataset_size:
                        break

        print(f"Built Dataset: found {self.__len__()} image-target pairs")

    def scale_and_pad(self, img, labels):
        scale_factor = self.max_h / max(*img.shape)

        scaled_img = skimage.transform.rescale(img, scale=scale_factor, multichannel=True)  # anti_aliasing=True

        img_h, img_w, _ = scaled_img.shape
        padded_scaled_img = np.zeros([self.max_h, self.max_w, 3])
        start_h, start_w = int((self.max_h - img_h) / 2), int((self.max_w - img_w) / 2)

        padded_scaled_img[start_h:start_h + img_h, start_w:start_w + img_w, :] = scaled_img
        padded_scaled_labels = (labels * scale_factor + np.array([[start_w], [start_h]])) / self.max_h - 0.5
        return padded_scaled_img, padded_scaled_labels

    def __getitem__(self, idx):
        return self.array_of_images[idx], self.array_of_labels[idx]

    def __len__(self):
        return self.array_of_images.shape[0]


if __name__ == "__main__":
    dataset = MPII_Dataset()
    print(dataset)
