import os
import copy
import logging
from tqdm import tqdm
from pathlib import Path

import json
import cv2
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
import albumentations as albu

from config import ParseConfig
from utils.pose import heatmap_generator


jnt_to_ind = {'head': 0, 'neck': 1, 'lsho': 2, 'lelb': 3, 'lwri': 4, 'rsho': 5, 'relb': 6, 'rwri': 7,
              'lhip': 8, 'lknee': 9, 'lankl': 10, 'rhip': 11, 'rknee': 12, 'rankl': 13}

ind_to_jnt = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
              8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}


##Add specific joints for coco dataset
coco_idx_to_jnt = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'lsho', 6: 'rsho',
                    7: 'lelb', 8: 'relb', 9: 'lwri', 10: 'rwri', 11: 'lhip', 12: 'rhip', 13: 'lknee',
                    14: 'rknee',15: 'lankl',16: 'rankl'}

coco_jnt_to_idx = {'nose':0,'left_eye':1, 'right_eye':2, 'left_ear':3, 'right_ear':4, 'lsho':5, 'rsho':6,
                    'lelb':7, 'relb':8, 'lwri':9, 'rwri':10, 'lhip':11, 'rhip':12, 'lknee':13,
                    'rknee':14,'lankl':15, 'rankl':16}

def load_coco(coco_conf: dict, load_images: bool = True) -> dict:
    """
    Converts Coco dataset to a dictionary object similar in structure as load_mpii.
    :param mpii_conf: Currently, determines whether to load mpii precached or not
    :param load_images: If RAM allows, load all images in memory
    :return: Dictionary containing the MPII dataset
    """
    root = Path(os.getcwd()).parent
    precached_coco = coco_conf['precached']

    # Load COCO images directly into memory
    if (precached_coco):

        logging.info('Loading precached COCO.')
        
        cached_path = '/mnt/vita/scratch/datasets/MS_COCO_rwx'
        img_dict = np.load(os.path.join(cached_path, 'cached', 'coco_cache_imgs_{}.npy'.format(load_images)), allow_pickle=True)#Use root if file in cached (with probabilistic pose)
        img_dict = img_dict[()]

        logging.info('Precached COCO loading succesfull.')
        return img_dict

    
    logging.info('Loading COCO without precached.')

    #dataset_path = os.path.join(root,'data', 'coco', 'annotations', 'person_keypoints_val2017.json')#Open the annotation file
    dataset_path_val = '/mnt/vita/scratch/datasets/MS_COCO_rwx/annotations/person_keypoints_val2017.json'    
    dataset_path_train = '/mnt/vita/scratch/datasets/MS_COCO_rwx/annotations/person_keypoints_train2017.json'    

    with open(dataset_path_val, 'r') as file:
        data_val = json.load(file)

    with open(dataset_path_train, 'r') as file:
        data_train = json.load(file)

    annotation_val = data_val['annotations']
    categories_val = data_val['categories']

    annotation_train = data_train['annotations']
    categories_train = data_train['categories']

    #image_folder_path = os.path.join(root,'data', 'coco', 'val2017')
    image_folder_path_val = '/mnt/vita/scratch/datasets/MS_COCO_rwx/images/val2017'
    image_folder_path_train = '/mnt/vita/scratch/datasets/MS_COCO_rwx/images/train2017'
    
    nbr_keypoints = len(categories_val[0]['keypoints'])
    assert(len(categories_val[0]['keypoints'])==len(categories_train[0]['keypoints'])), "not same number of keypoints"
    coco_template = dict([(coco_idx_to_jnt[i], []) for i in range(0,nbr_keypoints)])

    img_dict = {
        'coco': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [],
                 'dataset': [], 'num_gt': [], 'split': [], 'scale': [], 'objpos': [], 'num_ppl': []}}
    #In opposition to mpii, the normalizer is computed with the bounding box in create_coco


    #There is more annotations than images, so go through every images, then check for each image the different annotations corresponding to the image_id
    max_person_in_img = 0
    for filename in tqdm(os.listdir(image_folder_path_val) + os.listdir(image_folder_path_train)):
        if filename.endswith(('.jpg')): #Just to make sure
            if filename in os.listdir(image_folder_path_train):
                image_path = os.path.join(image_folder_path_train, filename)
                train_var = 1
            else:
                image_path = os.path.join(image_folder_path_val, filename)
                train_var = 0

            if load_images:
                loaded_image = cv2.imread(image_path)
                assert loaded_image.shape[2] == 3, "Image loaded with wrong dimensions" #Did this cause with plt.imread somehow one image had shape[2] = 1 (only Grayscale)
                img_dict['coco']['img'].append(loaded_image)

            image_id = int(filename[:-4])
            img_dict['coco']['img_name'].append(image_id)
            img_dict['coco']['dataset'].append('coco')

            num_people = 0
            num_keypoints = []
            gt_per_image = copy.deepcopy(coco_template)
            
            #Now checks when an image has annotation in the json files
            for image_special in (annotation_val+annotation_train):
                if image_id == image_special['image_id']:
                    for i in range(nbr_keypoints):
                        x,y,v = image_special['keypoints'][i*3:3*(i+1)]
                        if v < 0.5: #in coco 0 none, 1 occluded, 2 visible
                            x = -1
                            y = -1
                        v -= 1  #to have same format as MPII, LSP dataset: (-1,0,1) for  the flag
                        #if a point is not visible we also set (x,y) = (-1,-1)

                        gt_per_image[coco_idx_to_jnt[i]].append(np.array([x,y,v]))

                    num_keypoints.append(image_special['num_keypoints'])
                    num_people+=1

            max_person_in_img = max(max_person_in_img,num_people)
            img_dict['coco']['num_ppl'].append(num_people)
            img_dict['coco']['num_gt'].append(num_keypoints)
            img_dict['coco']['img_gt'].append(gt_per_image)

            #Set split based on if image comes from validation or training dataset
            if train_var ==1 : 
                split = 0 #Split = 0 => Train
            else:
                split = 1 #Split = 1 => Validation

            img_dict['coco']['split'].append(split)


    img_dict['coco']['max_person_in_img'] = max_person_in_img

    #Save for future cache
    cached_path = '/mnt/vita/scratch/datasets/MS_COCO_rwx'
    np.save(file=os.path.join(cached_path, 'cached', 'coco_cache_imgs_{}.npy'.format(load_images)),
            arr=img_dict, allow_pickle=True)#Use root if in cached (with probabilistic_pose)
    
    return img_dict

def load_mpii(mpii_conf: dict, load_images: bool = True) -> dict:
    """
    Converts Matlab structure .mat file into a more intuitive dictionary object.
    :param mpii_conf: Currently, determines whether to load mpii precached or not
    :param load_images: If RAM allows, load all images in memory
    :return: Dictionary containing the MPII dataset
    """

    # Lambda_head brings the head keypoint annotation closer to the neck
    lambda_head = 0.8
    precached_mpii = mpii_conf['precached']
    root = Path(os.getcwd()).parent
    dataset_path = '/mnt/vita/scratch/datasets/mpii'
    # Load MPII images directly into memory
    if precached_mpii:

        logging.info('Loading precached MPII.')
        
        img_dict = np.load(
            os.path.join(root, 'cached', 'mpii_cache_imgs_{}.npy'.format(load_images)), allow_pickle=True)
        img_dict = img_dict[()]

        try:
            assert img_dict['mpii']['lambda_head'] == lambda_head
            del img_dict['mpii']['lambda_head']

            return img_dict

        except AssertionError:
            logging.warning('Cannot load MPII due to different configurations.')
            logging.warning('Loading MPII from scratch.\n')


    mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                       6: 'pelvis', 7: 'thorax', 8: 'neck', 11: 'relb', 10: 'rwri', 9: 'head',
                       12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

    max_person_in_img = 0

    # Create a template for GT and Pred to follow
    mpii_template = dict([(mpii_idx_to_jnt[i], []) for i in range(16)])
    img_dict = {
        'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                 'dataset': [], 'num_gt': [], 'split': [], 'scale': [], 'objpos': [], 'num_ppl': []}}

    # Load MPII
    matlab_mpii = scipy.io.loadmat(
        os.path.join(dataset_path, 'joints.mat'), struct_as_record=False)['RELEASE'][0, 0]

    # Iterate over all images
    for img_idx in tqdm(range(matlab_mpii.__dict__['annolist'][0].shape[0]), ascii=True):

        # Load annotation data per image
        annotation_mpii = matlab_mpii.__dict__['annolist'][0, img_idx]
        train_test_mpii = matlab_mpii.__dict__['img_train'][0, img_idx].flatten()[0]

        person_id = matlab_mpii.__dict__['single_person'][img_idx][0].flatten()
        num_people = len(person_id)
        max_person_in_img = max(max_person_in_img, len(person_id))

        # Read image
        img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]

        try:
            image = plt.imread(os.path.join(dataset_path, 'images', img_name))
        except FileNotFoundError:
            logging.warning('Could not load filename: {}'.format(img_name))   ###WE ARE HERE HELLO--------------------------------------------------
            continue

        # Create a deepcopy of the template to avoid overwriting the original
        gt_per_image = copy.deepcopy(mpii_template)
        num_joints_persons = []
        normalizer_persons = []
        scale = []
        objpos = []

        # Default is that there are no annotated people in the image
        annotated_person_flag = False

        # Iterate over each person
        for person in (person_id - 1):
            try:
                per_person_jnts = []

                # If annopoints not present, then annotations for that person absent.
                # Throw exception and skip to next
                annopoints_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                scale_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['scale'][0][0]
                objpose_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['objpos'][0][0]
                objpose_img_mpii = [objpose_img_mpii.__dict__['x'][0][0], objpose_img_mpii.__dict__['y'][0][0]]
                num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]
                remove_pelvis_thorax_from_num_joints = 0

                # PCKh@0.x: Head bounding box normalizer
                head_x1 = annotation_mpii.__dict__['annorect'][0, person].__dict__['x1'][0][0]
                head_y1 = annotation_mpii.__dict__['annorect'][0, person].__dict__['y1'][0][0]
                head_x2 = annotation_mpii.__dict__['annorect'][0, person].__dict__['x2'][0][0]
                head_y2 = annotation_mpii.__dict__['annorect'][0, person].__dict__['y2'][0][0]
                xy_1 = np.array([head_x1, head_y1], dtype=np.float32)
                xy_2 = np.array([head_x2, head_y2], dtype=np.float32)

                normalizer_persons.append(np.linalg.norm(xy_1 - xy_2, ord=2))

                # If both are true, pulls the head joint closer to the neck, and body
                head_jt, neck_jt = False, False

                # MPII does not have a [-1, -1] or absent GT, hence the number of gt differ for each image
                for i in range(num_joints):
                    x = annopoints_img_mpii.__dict__['point'][0, i].__dict__['x'].flatten()[0]
                    y = annopoints_img_mpii.__dict__['point'][0, i].__dict__['y'].flatten()[0]
                    id_ = annopoints_img_mpii.__dict__['point'][0, i].__dict__['id'][0][0]
                    vis = annopoints_img_mpii.__dict__['point'][0, i].__dict__['is_visible'].flatten()

                    # No entry corresponding to visible, mostly head vis is missing.
                    if vis.size == 0:
                        vis = 1
                    else:
                        vis = vis.item()

                    if id_ == 9: head_jt = True
                    if id_ == 8: neck_jt = True

                    if ((id_ == 6) or (id_ == 7)):
                        remove_pelvis_thorax_from_num_joints += 1

                    # Arrange ground truth in form {jnt: [[person1], [person2]]}
                    gt_per_joint = np.array([x, y, vis]).astype(np.float16)
                    gt_per_image[mpii_idx_to_jnt[id_]].append(gt_per_joint)

                    per_person_jnts.append(mpii_idx_to_jnt[id_])

                # If person 1 lacks rankl and person 2 has rankl, then prevent rankl being associated with p1
                # If jnt absent in person, then we append np.array([-1, -1, -1])
                all_jnts = set(list(mpii_idx_to_jnt.values()))
                per_person_jnts = set(per_person_jnts)
                jnt_absent_person = all_jnts - per_person_jnts
                for abs_joint in jnt_absent_person:
                    gt_per_image[abs_joint].append(np.array([-1, -1, -1]))

                num_joints_persons.append(num_joints - remove_pelvis_thorax_from_num_joints)
                scale.append(scale_img_mpii)
                objpos.append(objpose_img_mpii)

                # If both head and neck joint present, then move the head joint linearly towards the neck joint.
                if head_jt and neck_jt:
                    gt_per_image['head'][-1] = (lambda_head * gt_per_image['head'][-1])\
                                               + ((1 - lambda_head) * gt_per_image['neck'][-1])

                # Since annotation for atleast on person in image present, this flag will add GT to the dataset
                annotated_person_flag = True

            except KeyError:
                # Person 'x' could not have annotated joints, hence move to person 'y'
                continue

        if not annotated_person_flag:
            continue

        # Maintain compatibility with MPII and LSPET
        del gt_per_image['pelvis']
        del gt_per_image['thorax']

        # Add image, name, pred placeholder and gt
        if load_images:
            img_dict['mpii']['img'].append(image)
        img_dict['mpii']['img_name'].append(img_name)
        img_dict['mpii']['img_pred'].append(mpii_template.copy())
        img_dict['mpii']['img_gt'].append(gt_per_image)
        img_dict['mpii']['normalizer'].append(normalizer_persons)
        img_dict['mpii']['dataset'].append('mpii')
        img_dict['mpii']['num_gt'].append(num_joints_persons)
        img_dict['mpii']['split'].append(train_test_mpii)
        img_dict['mpii']['scale'].append(scale)
        img_dict['mpii']['objpos'].append(objpos)
        img_dict['mpii']['num_ppl'].append(num_people)

    img_dict['mpii']['lambda_head'] = lambda_head
    img_dict['mpii']['max_person_in_img'] = max_person_in_img

    np.save(file=os.path.join(root, 'cached', 'mpii_cache_imgs_{}.npy'.format(load_images)),
            arr=img_dict, allow_pickle=True)

    del img_dict['mpii']['lambda_head']

    return img_dict


def load_lsp(load_images: bool = True) -> dict:
    """
    Load LSP dataset
    :return: Dictionary containing the LSP dataset
    """

    root = Path(os.getcwd()).parent
    dataset_path = '/mnt/vita/scratch/datasets/lsp'

    with open(os.path.join(dataset_path, 'lsp_filenames.txt'), 'r') as f:
        filenames = f.read().split()

    lsp_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip', 6: 'rwri', 7: 'relb',
                      8: 'rsho', 11: 'lwri', 10: 'lelb', 9: 'lsho', 12: 'neck', 13: 'head'}
    
    lsp_template = dict([(lsp_idx_to_jnt[i], []) for i in range(14)])
    img_dict = {'lsp': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                        'dataset': [], 'num_gt': [], 'split': []}}

    annotation_lsp = scipy.io.loadmat(os.path.join(dataset_path, 'joints.mat'))['joints']  # Shape: 3,14,2000

    # 0: Train; 1: Validate
    # Load Train/Test split if conf.model_load_hg == True
    train_test_split = np.concatenate(
        [np.zeros((int(annotation_lsp.shape[2] * 0.5),), dtype=np.int8), # Half is for training, other half is for validation
         np.ones((annotation_lsp.shape[2] - int(annotation_lsp.shape[2] * 0.5),), dtype=np.int8)],
        axis=0)

    # Create common dictionary format for DataLoader to access
    for index in tqdm(range(annotation_lsp.shape[2]), ascii=True):
        image = plt.imread(os.path.join(dataset_path, 'images', filenames[index]))

        # Broadcasting rules apply: Toggle visibility of ground truth
        gt = abs(np.array([[0], [0], [1]]) - annotation_lsp[:, :, index])
        gt_dict = dict([(lsp_idx_to_jnt[i], [gt[:, i]]) for i in range(gt.shape[1])])
        num_gt = sum([1 for i in range(gt.shape[1]) if gt[:, i][2]])

        # PCK@0.x : Normalizer
        lsho = gt[:2, 9]
        rsho = gt[:2, 8]
        lhip = gt[:2, 3]
        rhip = gt[:2, 2]
        torso_1 = np.linalg.norm(lsho - rhip)
        torso_2 = np.linalg.norm(rsho - lhip)
        torso = max(torso_1, torso_2)

        if load_images:
            img_dict['lsp']['img'].append(image)
        img_dict['lsp']['img_name'].append(filenames[index])
        img_dict['lsp']['img_pred'].append(copy.deepcopy(lsp_template))
        img_dict['lsp']['img_gt'].append(gt_dict)
        img_dict['lsp']['normalizer'].append([torso])
        img_dict['lsp']['dataset'].append('lsp')
        img_dict['lsp']['num_gt'].append([num_gt])
        img_dict['lsp']['split'].append(train_test_split[index])

    return img_dict


def load_lspet(load_images: bool = True) -> dict:
    """
    Load LSP dataset
    :return: Dictionary containing the LSP dataset
    """

    root = Path(os.getcwd()).parent
    dataset_path = '/mnt/vita/scratch/datasets/lspet'

    with open(os.path.join(dataset_path, 'lspet_filenames.txt'), 'r') as f:
        filenames = f.read().split()

    lspet_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                        6: 'rwri', 7: 'relb', 8: 'rsho', 11: 'lwri', 10: 'lelb', 9: 'lsho',
                        12: 'neck', 13: 'head'}

    lspet_template = dict([(lspet_idx_to_jnt[i], []) for i in range(14)])

    img_dict = {'lspet': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                          'dataset': [], 'num_gt': [], 'split': []}}

    annotation_lspet = scipy.io.loadmat(os.path.join(dataset_path, 'joints.mat'))['joints']  # Shape: 14,3,10000

    # 0: Train; 1: Validate
    # All of LSPET is a part of the training dataset
    train_test_split = np.concatenate(
        [np.zeros((annotation_lspet.shape[2],), dtype=np.int8),
         np.ones((annotation_lspet.shape[2] - annotation_lspet.shape[2],), dtype=np.int8)],
        axis=0)

    
    # Create common dictionary format for DataLoader to access
    for index in tqdm(range(annotation_lspet.shape[2]), ascii=True):
        image = plt.imread(os.path.join(dataset_path, 'images', filenames[index]))
        gt = annotation_lspet[:, :, index]
        gt_dict = dict([(lspet_idx_to_jnt[i], [gt[i]]) for i in range(gt.shape[0])])
        num_gt = sum([1 for i in range(gt.shape[0]) if gt[i][2]])

        # PCK@0.x : Normalizer
        lsho = gt[9, :2]
        rsho = gt[8, :2]
        lhip = gt[3, :2]
        rhip = gt[2, :2]
        torso_1 = np.linalg.norm(lsho - rhip)
        torso_2 = np.linalg.norm(rsho - lhip)
        torso = max(torso_1, torso_2)

        if load_images:
            img_dict['lspet']['img'].append(image)
        img_dict['lspet']['img_name'].append(filenames[index])
        img_dict['lspet']['img_pred'].append(copy.deepcopy(lspet_template))
        img_dict['lspet']['img_gt'].append(gt_dict)
        img_dict['lspet']['normalizer'].append([torso])
        img_dict['lspet']['dataset'].append('lspet')
        img_dict['lspet']['num_gt'].append([num_gt])
        img_dict['lspet']['split'].append(train_test_split[index])

    return img_dict


def load_hp_dataset(dataset_conf: dict, load_images: bool = True, load_full: bool = True) -> dict:
    """
    Loads MPII and LSP-LSPET from native format to numpy
    """

    #Loading both datasets if necessary
    dataset_dict = dict()

    if dataset_conf['load']=='coco' or dataset_conf['load']=='merged':
        if (not load_full):
            logging.info('Loading COCO dataset')
            dataset_dict.update(load_coco(coco_conf=dataset_conf['coco_params'], load_images=load_images))

    if dataset_conf['load']=='mpii' or dataset_conf['load']=='merged':
        logging.info('Loading MPII dataset')
        dataset_dict.update(load_mpii(mpii_conf=dataset_conf['mpii_params'], load_images=load_images))

        logging.info('Loading LSP dataset')
        dataset_dict.update(load_lsp(load_images=load_images))
        
        logging.info('Loading LSPET dataset')
        dataset_dict.update(load_lspet(load_images=load_images))
    
    return dataset_dict



class HumanPoseDataLoader(torch.utils.data.Dataset):

    def __init__(self, dataset_dict: dict, conf: ParseConfig) -> None:
        """
        Sampler for MPII and LSP-LSPET datasets
        :param dataset_dict: Dictionary containing MPII and LSP-LSPET numpy arrays
        :param conf: Configuration for the experiment
        """

        self.conf = conf
        self.occlusion = True
        self.hm_shape = (64, 64)
        self.hm_peak = 30
        self.model_save_path = conf.save_path
        self.load_images = conf.load_images
        self.load_full = conf.load_full

        self.jnt_to_ind = jnt_to_ind
        self.ind_to_jnt = ind_to_jnt

        #Saw that is was used in the rest of the code (Main)
        self.jnt_to_ind = coco_jnt_to_idx
        self.ind_to_jnt = coco_idx_to_jnt
        self.coco_idx_to_jnt = coco_idx_to_jnt
        self.coco_jnt_to_idx = coco_jnt_to_idx

        # Training specific attributes:
        self.model_input_dataset = None

        self.augmentation_flag = False

        self.dataset_size = dict()

        # Create dataset by converting into standard format
        if self.conf.dataset['load']=='coco' or self.conf.dataset['load']=='merged':
            if (self.load_full):
                #If want to use data with only all_joints and load them directly, change coco_full_train_not_all_joints to coco_full_train_all_joints, same dor test
                logging.info('Loading Precached train and validation set\n')
                cached_path = '/mnt/vita/scratch/datasets/MS_COCO_rwx'
                train_dict = np.load(file=os.path.join(cached_path, 'cached', 'coco_full_train_not_all_joints.npy'), allow_pickle=True)#Use root if file in cached (with probabilistic pose)
                self.coco_train = train_dict[()]
                val_dict = np.load(file=os.path.join(cached_path, 'cached', 'coco_full_validation_not_all_joints.npy'), allow_pickle=True)#Use root if file in cached (with probabilistic pose)
                self.coco_validate = val_dict[()]
                self.coco = {}
                self.coco['max_person_in_img'] = 20
                logging.info('Done loading Precached train and validation set\n')
                logging.info('\nSize of train and validation dataset after loading: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))

                if self.conf.dataset['coco_params']['all_joints'] == False:
                    logging.info('Treating data for enough gt per image and bbox size\n')
                    self.coco_train = self.coco_num_gt(coco_dataset=self.coco_train)
                    self.coco_validate = self.coco_num_gt(coco_dataset=self.coco_validate)
                    logging.info('Done treating data for enough gt per image\n')
                    self.coco_train = self.coco_bbox_size(coco_dataset=self.coco_train)
                    self.coco_validate = self.coco_bbox_size(coco_dataset=self.coco_validate)
                    logging.info('\nFinal size of train and validation dataset after bbox size selection: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))
                    logging.info('Done treating data for bbox size\n')
                else:
                    assert self.conf.dataset['coco_params']['all_joints'] == True
                    logging.info('Selecting train and validation images where all joints are present.\n')
                    self.coco_train = self.coco_all_joints(coco_dataset=self.coco_train)
                    self.coco_validate = self.coco_all_joints(coco_dataset=self.coco_validate)
                    logging.info('\nFinal size of train and validation dataset after all joints selection: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))
                    logging.info('Done treating data for all_joints present\n')

                logging.info('\nFinal size of train and validation dataset after selection: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))


            else:
                logging.info('Creating COCO dataset\n')
                self.coco = dataset_dict['coco']
                self.coco_dataset = self.create_coco()
                self.coco_train, self.coco_validate = self.create_train_validate(dataset=self.coco_dataset)
                logging.info('\nFinal size of train and validation dataset: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))

                logging.info('Creating single person patches\n')
                self.coco_train = self.coco_single_person_extractor(coco_dataset=self.coco_train)
                self.coco_validate = self.coco_single_person_extractor(coco_dataset=self.coco_validate)
                logging.info('\nFinal size of train and validation dataset after single person patches: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))

                ### Choose your functions to treat the data

                #logging.info('Selecting train and validation images where all joints are present.\n')
                #self.coco_train = self.coco_all_joints(coco_dataset=self.coco_train, train = 1)
                #self.coco_validate = self.coco_all_joints(coco_dataset=self.coco_validate, train = 0)
                #logging.info('\nFinal size of train and validation dataset after all joints selection: {} and {}'.format(self.coco_train['name'].shape[0],self.coco_validate['name'].shape[0]))

            self.dataset_size.update(
                {'coco': {'train': self.coco_train['name'].shape[0],
                        'validation': self.coco_validate['name'].shape[0]}})
            if self.conf.dataset['load']=='coco':
                self.model_input_dataset = self.merge_dataset(datasets=[self.coco_train, self.coco_validate], indices=[np.arange(self.coco_train['name'].shape[0]),np.arange(self.coco_validate['name'].shape[0])])
                self.train = self.merge_dataset(datasets=[self.coco_train],indices=[np.arange(self.coco_train['name'].shape[0])])
                self.validate = self.merge_dataset(datasets=[self.coco_validate],indices=[np.arange(self.coco_validate['name'].shape[0])])

                del self.coco_train, self.coco_validate
        
        if self.conf.dataset['load']=='mpii' or self.conf.dataset['load']=='merged':
            # Create dataset by converting into standard format
            logging.info('Creating MPII dataset\n')
            self.mpii = dataset_dict['mpii']
            self.mpii_dataset = self.create_mpii()
            self.mpii_train, self.mpii_validate = self.create_train_validate(dataset=self.mpii_dataset)

            logging.info('Creating single person patches\n')
            self.mpii_train = self.mpii_single_person_extractor(mpii_dataset=self.mpii_train)
            self.mpii_validate = self.mpii_single_person_extractor(mpii_dataset=self.mpii_validate)

            logging.info('Selecting train and validation images where all joints are present.\n')
            self.mpii_train = self.mpii_all_joints(mpii_dataset=self.mpii_train)
            self.mpii_validate = self.mpii_all_joints(mpii_dataset=self.mpii_validate)

            logging.info('Size of MPII processed dataset: ')
            logging.info('Train: {}'.format(self.mpii_train['name'].shape[0]))
            logging.info('Validate: {}\n'.format(self.mpii_validate['name'].shape[0]))

            self.dataset_size.update(
                {'mpii': {'train': self.mpii_train['name'].shape[0],
                        'validation': self.mpii_validate['name'].shape[0]}})

            #Add the lspet and lsp dataset
            self.lspet = dataset_dict['lspet']
            self.lsp = dataset_dict['lsp']

            logging.info('Creating LSPET dataset\n')
            self.lspet_dataset = self.create_lspet()
                
            logging.info('Creating LSP dataset\n')
            self.lsp_dataset = self.create_lsp()
                
            self.lspet_train, self.lspet_validate = self.create_train_validate(dataset=self.lspet_dataset)
            self.lsp_train, self.lsp_validate = self.create_train_validate(dataset=self.lsp_dataset)

            self.dataset_size.update({'lspet': {'train': self.lspet_train['name'].shape[0],'validation': self.lspet_validate['name'].shape[0]},'lsp': {'train': self.lsp_train['name'].shape[0],'validation': self.lsp_validate['name'].shape[0]}})

            if self.conf.dataset['load']=='mpii':
                self.model_input_dataset = self.merge_dataset(datasets=[self.mpii_train,
                                                                        self.lspet_train,
                                                                        self.lsp_train,
                                                                        self.mpii_validate,
                                                                        self.lspet_validate,
                                                                        self.lsp_validate],
                                                            indices=[np.arange(self.mpii_train['name'].shape[0]),
                                                                     np.arange(self.lspet_train['name'].shape[0]),
                                                                     np.arange(self.lsp_train['name'].shape[0]),
                                                                     np.arange(self.mpii_validate['name'].shape[0]),
                                                                     np.arange(self.lspet_validate['name'].shape[0]),
                                                                     np.arange(self.lsp_validate['name'].shape[0])])
                self.train = self.merge_dataset(datasets=[self.mpii_train, self.lspet_train, self.lsp_train],indices=[np.arange(self.mpii_train['name'].shape[0]),np.arange(self.lspet_train['name'].shape[0]),np.arange(self.lsp_train['name'].shape[0])])
                self.validate = self.merge_dataset(datasets=[self.mpii_validate,self.lspet_validate, self.lsp_validate],indices=[np.arange(self.mpii_validate['name'].shape[0]),np.arange(self.lspet_validate['name'].shape[0]),np.arange(self.lsp_validate['name'].shape[0])])
                del self.mpii_train, self.mpii_validate, self.lspet_train,self.lspet_validate,self.lsp_train,self.lsp_validate

        if self.conf.dataset['load']=='merged':
            assert self.conf.dataset['load'] == 'merged', "dataset['load'] should be mpii, coco or merged"
            self.model_input_dataset = self.merge_dataset(datasets=[self.coco_train,
                            self.mpii_train,
                            self.lspet_train,
                            self.lsp_train,
                            self.coco_validate,
                            self.mpii_validate,
                            self.lspet_validate,
                            self.lsp_validate],
                    indices=[np.arange(self.coco_train['name'].shape[0]),
                            np.arange(self.mpii_train['name'].shape[0]),
                            np.arange(self.lspet_train['name'].shape[0]),
                            np.arange(self.lsp_train['name'].shape[0]),
                            np.arange(self.coco_validate['name'].shape[0]), 
                            np.arange(self.mpii_validate['name'].shape[0]),
                            np.arange(self.lspet_validate['name'].shape[0]),
                            np.arange(self.lsp_validate['name'].shape[0])])
            self.train = self.merge_dataset(datasets=[self.coco_train, self.mpii_train, self.lspet_train, self.lsp_train],indices=[np.arange(self.coco_train['name'].shape[0]),np.arange(self.mpii_train['name'].shape[0]),np.arange(self.lspet_train['name'].shape[0]),np.arange(self.lsp_train['name'].shape[0])])
            self.validate = self.merge_dataset(datasets=[self.coco_validate,self.mpii_validate,self.lspet_validate, self.lsp_validate],indices=[np.arange(self.coco_validate['name'].shape[0]),np.arange(self.mpii_validate['name'].shape[0]),np.arange(self.lspet_validate['name'].shape[0]),np.arange(self.lsp_validate['name'].shape[0])])
            del self.mpii_train, self.mpii_validate, self.coco_train, self.coco_validate, self.lspet_train,self.lspet_validate,self.lsp_train,self.lsp_validate
        else:
            assert self.conf.dataset['load'] == 'coco' or self.conf.dataset['load'] == 'mpii', "dataset['load'] should be mpii, coco or merged"

        logging.info('\nFinal size of dataset: {}'.format(self.model_input_dataset['name'].shape[0]))

        # Deciding augmentation techniques
        self.shift_scale_rotate = self.augmentation(
            [albu.ShiftScaleRotate(
                p=1, shift_limit=0.2, scale_limit=0.25, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, value=0)
            ])

        self.flip_prob = 0.5
        self.horizontal_flip = self.augmentation([albu.HorizontalFlip(p=1)])

        logging.info('\nDataloader Initialized.\n')


    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return self.model_input_dataset['gt'].shape[0]


    def __getitem__(self, i: int) -> (torch.Tensor, torch.Tensor):
        """
        Returns images and heatmaps from the dataset.
        """
        root = Path(os.getcwd()).parent
        mpii_path = os.path.join(root, 'data', 'mpii')
        lsp_path = os.path.join(root, 'data', 'lsp')
        lspet_path = os.path.join(root, 'data', 'lspet')
        coco_path = os.path.join(root, 'data', 'coco')

        name = self.model_input_dataset['name'][i]
        gt = self.model_input_dataset['gt'][i]
        dataset = self.model_input_dataset['dataset'][i]
        num_gt = self.model_input_dataset['num_gt'][i]
        split = self.model_input_dataset['split'][i]
        num_persons = self.model_input_dataset['num_persons'][i]
        bbox_coords = self.model_input_dataset['bbox_coords'][i]
        normalizer = self.model_input_dataset['normalizer'][i]

        if self.load_images:
            image = self.model_input_dataset['img'][i]
            assert image.shape[2] == 3, 'Image loaded with wrong dimensions got image shape: {}'.format(image.shape)

        else:
            if dataset == 'mpii':
                image = plt.imread(os.path.join(mpii_path, 'images', '{}.jpg'.format(name.split('_')[0])))
            elif dataset == 'lsp':
                image = plt.imread(os.path.join(lsp_path, 'images', name))
            elif dataset == 'coco': #Added coco line

                filename = '{:012d}.jpg'.format(name)
                image_folder_path_val = '/mnt/vita/scratch/datasets/MS_COCO_rwx/images/val2017'
                image_folder_path_train = '/mnt/vita/scratch/datasets/MS_COCO_rwx/images/train2017'

                if filename in os.listdir(image_folder_path_train):
                    image = cv2.imread(os.path.join(image_folder_path_train, filename))
                else:
                    image = cv2.imread(os.path.join(image_folder_path_val, filename))

            else:
                image = plt.imread(os.path.join(lspet_path, 'images', name))

        # Convert from XY cartesian to UV image coordinates
        xy_to_uv = lambda xy: (xy[1], xy[0])

        # Determine crop
        img_shape = np.array(image.shape)
        [min_x, min_y, max_x, max_y] = bbox_coords[0]

        tl_uv = xy_to_uv(np.array([min_x, min_y]))
        br_uv = xy_to_uv(np.array([max_x, max_y]))
        min_u = tl_uv[0]
        min_v = tl_uv[1]
        max_u = br_uv[0]
        max_v = br_uv[1]

        centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        height = max_u - min_u
        width = max_v - min_v

        if self.augmentation_flag:
            scale = np.random.uniform(low=1.5,high=2)
        else:
            scale = 1.75

        window = max(scale * height, scale * width)
        top_left = np.array([centre[0] - (window / 2), centre[1] - (window / 2)])
        bottom_right = np.array([centre[0] + (window / 2), centre[1] + (window / 2)])

        top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # Cropping the image and adjusting the ground truth
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        for person in range(gt.shape[0]):
            for joint in range(gt.shape[1]):
                gt_uv = xy_to_uv(gt[person][joint])
                gt_uv = gt_uv - top_left
                gt[person][joint] = np.concatenate([gt_uv, np.array([gt[person][joint][2]])], axis=0)

        # Resize the image
        image, gt, scale_params = self.resize_image(image, gt, target_size=[256, 256, 3])

        # Augmentation
        if self.augmentation_flag:

            # Horizontal flip can't be done using albu's probability
            if torch.rand(1) < self.flip_prob:
                # Augment image and keypoints
                augmented = self.horizontal_flip(image=image, keypoints=gt.reshape(-1, 3)[:, :2])
                image = augmented['image']
                gt[:, :, :2] = np.stack(augmented['keypoints'], axis=0).reshape(-1, self.conf.experiment_settings['num_hm'], 2)

                if dataset == 'coco':
                    # Flip ground truth to match horizontal flip
                    gt[:, [coco_jnt_to_idx['lsho'], coco_jnt_to_idx['rsho']], :] = gt[
                        :, [coco_jnt_to_idx['rsho'], coco_jnt_to_idx['lsho']], :]
                    gt[:, [coco_jnt_to_idx['lelb'], coco_jnt_to_idx['relb']], :] = gt[
                        :, [coco_jnt_to_idx['relb'], coco_jnt_to_idx['lelb']], :]
                    gt[:, [coco_jnt_to_idx['lwri'], coco_jnt_to_idx['rwri']], :] = gt[
                        :, [coco_jnt_to_idx['rwri'], coco_jnt_to_idx['lwri']], :]
                    gt[:, [coco_jnt_to_idx['lhip'], coco_jnt_to_idx['rhip']], :] = gt[
                        :, [coco_jnt_to_idx['rhip'], coco_jnt_to_idx['lhip']], :]
                    gt[:, [coco_jnt_to_idx['lknee'], coco_jnt_to_idx['rknee']], :] = gt[
                        :, [coco_jnt_to_idx['rknee'], coco_jnt_to_idx['lknee']], :]
                    gt[:, [coco_jnt_to_idx['lankl'], coco_jnt_to_idx['rankl']], :] = gt[
                        :, [coco_jnt_to_idx['rankl'], coco_jnt_to_idx['lankl']], :]
                    gt[:, [coco_jnt_to_idx['left_eye'], coco_jnt_to_idx['right_eye']], :] = gt[
                        :, [coco_jnt_to_idx['right_eye'], coco_jnt_to_idx['left_eye']], :]
                    gt[:, [coco_jnt_to_idx['left_ear'], coco_jnt_to_idx['right_ear']], :] = gt[
                        :, [coco_jnt_to_idx['right_ear'], coco_jnt_to_idx['left_ear']], :]

                else:
                    # Flip ground truth to match horizontal flip
                    gt[:, [jnt_to_ind['lsho'], jnt_to_ind['rsho']], :] = gt[
                        :, [jnt_to_ind['rsho'], jnt_to_ind['lsho']], :]
                    gt[:, [jnt_to_ind['lelb'], jnt_to_ind['relb']], :] = gt[
                        :, [jnt_to_ind['relb'], jnt_to_ind['lelb']], :]
                    gt[:, [jnt_to_ind['lwri'], jnt_to_ind['rwri']], :] = gt[
                        :, [jnt_to_ind['rwri'], jnt_to_ind['lwri']], :]
                    gt[:, [jnt_to_ind['lhip'], jnt_to_ind['rhip']], :] = gt[
                        :, [jnt_to_ind['rhip'], jnt_to_ind['lhip']], :]
                    gt[:, [jnt_to_ind['lknee'], jnt_to_ind['rknee']], :] = gt[
                        :, [jnt_to_ind['rknee'], jnt_to_ind['lknee']], :]
                    gt[:, [jnt_to_ind['lankl'], jnt_to_ind['rankl']], :] = gt[
                        :, [jnt_to_ind['rankl'], jnt_to_ind['lankl']], :]


            # Ensure shift scale rotate augmentation retains all joints
            tries = 5
            augment_ok = False
            image_, gt_ = None, None

            while tries > 0:
                tries -= 1
                augmented = self.shift_scale_rotate(image=image, keypoints=gt.reshape(-1, 3)[:, :2])
                image_ = augmented['image']
                gt_ = np.stack(augmented['keypoints'], axis=0).reshape(
                    -1, self.conf.experiment_settings['num_hm'], 2)

                # I don't remember why I set the threshold to -+5 but I don't want to break it
                if (np.all(gt_[0]) > -5) and (np.all(gt_[0]) < 261):  # 0 index single person
                    augment_ok = True
                    break

            if augment_ok:
                image = image_
                gt[:, :, :2] = gt_

        heatmaps, _ = heatmap_generator(
            joints=np.copy(gt), occlusion=self.occlusion, hm_shape=self.hm_shape, img_shape=image.shape)

        heatmaps = self.hm_peak * heatmaps

        return torch.tensor(data=image/256.0, dtype=torch.float32, device='cpu'), \
               torch.tensor(data=heatmaps, dtype=torch.float32, device='cpu'),gt[:,:,:]
    
    def create_coco(self) -> dict:
        """
        Standardizes coco_dict into a common format for sampling
        """
        coco = self.coco
        max_persons = coco['max_person_in_img']
        len_dataset = len(coco['img_name'])

        assert len(coco['img_name']) == len(coco['img_gt']), "MPII dataset image and labels mismatched."

        # What the create coco dataset will look like.
        dataset = {
            'img': [],
            'name': [],
            'gt': -np.ones(shape=(len_dataset, max_persons, len(coco_jnt_to_idx), 3)),
            'dataset': [],
            'num_gt': np.zeros(shape=(len_dataset, max_persons)),
            'split': [],
            'num_persons': np.zeros(shape=(len_dataset, 1)),
            'normalizer': np.zeros(shape=(len_dataset, max_persons)),
            'bbox_coords': -np.ones(shape=(len_dataset, max_persons, 4))
        }

        
        split_array = [] #To store the different split values
        for i in tqdm(range(len_dataset)):
            if self.load_images:
                image = coco['img'][i]
            image_name = coco['img_name'][i]
            ground_truth = coco['img_gt'][i]
            dataset_ = coco['dataset'][i]
            num_gt = coco['num_gt'][i]
            split = coco['split'][i]


            # Computing the number of people
            num_ppl = 0
            for key in ground_truth.keys():
                num_ppl = max(num_ppl, len(ground_truth[key]))
                break   # All keys have same length, when jnt absent, then we append np.array([-1, -1, -1])
            dataset['num_persons'][i] = num_ppl
        
            # Split 0 indicates training dataset, split 1 indicates validation dataset
            assert split == 0 or split == 1, "All annotated images should have split == 1 or split == 0"

            # Assigning to a Numpy Ground truth array for every coco joint
            for jnt in ground_truth.keys():
                for person_id in range(len(ground_truth[jnt])):
                    dataset['gt'][i, person_id, coco_jnt_to_idx[jnt]] = ground_truth[jnt][person_id]   

            # Assigning Bounding Box coordinates per person
            for person_id in range(num_ppl):
                x_coord = dataset['gt'][i, person_id, :, 0][np.where(dataset['gt'][i, person_id, :, 0] > -0.5)]
                y_coord = dataset['gt'][i, person_id, :, 1][np.where(dataset['gt'][i, person_id, :, 1] > -0.5)]

                #If person has no visible coordinates we set the bounding box to (0,0)
                if x_coord.size > 0:
                    min_x = np.min(x_coord)
                    max_x = np.max(x_coord)
                else:
                    min_x = 0
                    max_x = 0 #np.inf

                if y_coord.size > 0:
                    min_y = np.min(y_coord)
                    max_y = np.max(y_coord)
                else:
                    min_y = 0
                    max_y = 0 #np.inf

                dataset['bbox_coords'][i, person_id] = np.array([min_x, min_y, max_x, max_y])

                #Compute the normalizer to be the diagonal of the bounding box
                xy_1 = np.array([max_x, max_y], dtype=np.float32)
                xy_2 = np.array([min_x, min_y], dtype=np.float32)
                normalizer_persons = np.linalg.norm(xy_1 - xy_2, ord=2)
                dataset['normalizer'][i, person_id] = normalizer_persons

            # Number of joints scaling factor
            for person_id in range(len(num_gt)):
                dataset['num_gt'][i, person_id] = num_gt[person_id]


            if self.load_images:
                assert image.shape[2] == 3, "Image loaded with wrong dimensions"
                dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            split_array.append(split)

        dataset['split'] = np.array([1 if x > 0.5 else 0 for x in split_array])

        dataset['img'] = np.array(dataset['img'], dtype=object) # array of shape == 0, dataset['img'] exists for legacy reasons only
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])

        logging.info('COCO dataset description:')
        logging.info('Length (#images): {}'.format(dataset['gt'].shape[0]))

        return dataset
    


    def create_mpii(self) -> dict:
        """
        Standardizes mpii_dict into a common format for sampling
        """

        mpii = self.mpii
        try:
            max_persons = self.coco['max_person_in_img']
            self.mpii['max_person_in_img'] = self.coco['max_person_in_img']
        except AttributeError:
            max_persons = mpii['max_person_in_img']

        len_dataset = len(mpii['img_name'])
        assert len(mpii['img_name']) == len(mpii['img_gt']), "MPII dataset image and labels mismatched."

        # What the create mpii dataset will look like.
        dataset = {
            'img': [],
            'name': [],
            'gt': -np.ones(shape=(len_dataset, max_persons, self.conf.experiment_settings['num_hm'], 3)),
            'dataset': [],
            'num_gt': np.zeros(shape=(len_dataset, max_persons)),
            'split': [],
            'num_persons': np.zeros(shape=(len_dataset, 1)),
            'normalizer': np.zeros(shape=(len_dataset, max_persons)),
            'bbox_coords': -np.ones(shape=(len_dataset, max_persons, 4))
        }

        # Converting the dataset to numpy arrays
        for i in range(len_dataset):

            if self.load_images:
                image = mpii['img'][i]
            image_name = mpii['img_name'][i]
            ground_truth = mpii['img_gt'][i]
            dataset_ = mpii['dataset'][i]
            num_gt = mpii['num_gt'][i]
            split = mpii['split'][i]
            normalizer = mpii['normalizer'][i]

            # Calculating the number of people
            num_ppl = 0
            for key in ground_truth.keys():
                num_ppl = max(num_ppl, len(ground_truth[key]))
                break   # All keys have same length, when jnt absent, then we append np.array([-1, -1, -1])
            dataset['num_persons'][i] = num_ppl

            # Split 0 indicates testing dataset
            assert split == 1, "All annotated images should have split == 1"

            # Assigning to a Numpy Ground truth array
            for jnt in ground_truth.keys():
                for person_id in range(len(ground_truth[jnt])):
                    dataset['gt'][i, person_id, jnt_to_ind[jnt]] = ground_truth[jnt][person_id]

            # Assigning Bounding Box coordinates per person
            for person_id in range(num_ppl):
                x_coord = dataset['gt'][i, person_id, :, 0][np.where(dataset['gt'][i, person_id, :, 0] > -1)]
                y_coord = dataset['gt'][i, person_id, :, 1][np.where(dataset['gt'][i, person_id, :, 1] > -1)]
                min_x = np.min(x_coord)
                max_x = np.max(x_coord)
                min_y = np.min(y_coord)
                max_y = np.max(y_coord)
                dataset['bbox_coords'][i, person_id] = np.array([min_x, min_y, max_x, max_y])

            # Number of joints scaling factor
            for person_id in range(len(num_gt)):
                dataset['num_gt'][i, person_id] = num_gt[person_id]

            # PCK normalizer
            for person_id in range(len(normalizer)):
                dataset['normalizer'][i, person_id] = normalizer[person_id]

            if self.load_images:
                dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        # Load Train/Test split if conf.model_load_hg = True
        root = Path(os.getcwd()).parent
        logging.info('\nCreating the Newell validation split.\n')
        with open(os.path.join(root, 'cached', 'Stacked_HG_ValidationImageNames.txt')) as valNames:
             valNames_ = [x.strip('\n') for x in valNames.readlines()]

        dataset['split'] = np.array([1 if x in valNames_ else 0 for x in dataset['name']])

        dataset['img'] = np.array(dataset['img'], dtype=object) # array of shape == 0, dataset['img'] exists for legacy reasons only
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])

        logging.info('MPII dataset description:')
        logging.info('Length (#images): {}'.format(dataset['gt'].shape[0]))

        return dataset


    def create_lspet(self) -> dict:
        """
        Standardizes lspet_dict into a common format for sampling
        """

        try:
            max_persons = self.mpii['max_person_in_img']
        except AttributeError:
            max_persons = 1

        lspet = copy.deepcopy(self.lspet)
        assert len(lspet['img_name']) == len(lspet['img_gt']), "LSPET dataset image and labels mismatched."

        dataset = {
            'img': [],
            'name': [],
            'gt': -np.ones(shape=(len(lspet['img_name']), max_persons, self.conf.experiment_settings['num_hm'], 3)),
            'dataset': [],
            'num_gt': np.zeros(shape=(len(lspet['img_name']), max_persons)),
            'split': [],
            'num_persons': np.ones(shape=(len(lspet['img_name']), 1)),
            'normalizer': np.zeros(shape=(len(lspet['img_name']), max_persons)),
            'bbox_coords': -np.ones(shape=(len(lspet['img_name']), max_persons, 4))
        }    # max_persons is always 1 for lsp*

        len_dataset = len(lspet['img_name'])
        for i in range(len_dataset):
            if self.load_images:
                image = lspet['img'][i]
            image_name = lspet['img_name'][i]
            ground_truth = lspet['img_gt'][i]
            dataset_ = lspet['dataset'][i]
            num_gt = lspet['num_gt'][i]
            split = lspet['split'][i]
            normalizer = lspet['normalizer'][i]

            for jnt in ground_truth.keys():
                dataset['gt'][i, 0, jnt_to_ind[jnt]] = ground_truth[jnt][0]

            # Assigning Bounding Box coordinates per person
            x_coord = dataset['gt'][i, 0, :, 0][np.where(dataset['gt'][i, 0, :, 2] == 1)]
            y_coord = dataset['gt'][i, 0, :, 1][np.where(dataset['gt'][i, 0, :, 2] == 1)]

            x_coord = x_coord[np.where(x_coord > -1)]
            y_coord = y_coord[np.where(y_coord > -1)]

            min_x = np.min(x_coord)
            max_x = np.max(x_coord)
            min_y = np.min(y_coord)
            max_y = np.max(y_coord)

            dataset['bbox_coords'][i, 0] = np.array([min_x, min_y, max_x, max_y])

            # Assigning number of GT to person 0
            dataset['num_gt'][i, 0] = num_gt[0]

            dataset['normalizer'][i, 0] = normalizer[0]
            
            if self.load_images:
                dataset['img'].append(image)	
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        dataset['img'] = np.array(dataset['img'], dtype=object)        # Empty array if load_all_images_in_memory = False
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        logging.info('LSPET dataset description:')
        logging.info('Length (#images): {}'.format(dataset['gt'].shape[0]))

        return dataset


    def create_lsp(self) -> dict:
        """
        Standardizes lsp_dict into a common format for sampling
        """

        try:
            max_persons = self.mpii['max_person_in_img']
        except AttributeError:
            max_persons = 1

        lsp = copy.deepcopy(self.lsp)
        assert len(lsp['img_name']) == len(lsp['img_gt']), "LSP dataset image and labels mismatched."

        dataset = {
            'img': [],
            'name': [],
            'gt': -np.ones(shape=(len(lsp['img_name']), max_persons, self.conf.experiment_settings['num_hm'], 3)),
            'dataset': [],
            'num_gt': np.zeros(shape=(len(lsp['img_name']), max_persons)),
            'split': [],
            'num_persons': np.ones(shape=(len(lsp['img_name']), 1)),
            'normalizer': np.zeros(shape=(len(lsp['img_name']), max_persons)),
            'bbox_coords': -np.ones(shape=(len(lsp['img_name']), max_persons, 4))
        }  # max_persons is always 1 for lsp*

        len_dataset = len(lsp['img_name'])
        for i in range(len_dataset):

            if self.load_images:
                image = lsp['img'][i]
            image_name = lsp['img_name'][i]
            ground_truth = lsp['img_gt'][i]
            dataset_ = lsp['dataset'][i]
            num_gt = lsp['num_gt'][i]
            split = lsp['split'][i]
            normalizer = lsp['normalizer'][i]

            for jnt in ground_truth.keys():
                dataset['gt'][i, 0, jnt_to_ind[jnt]] = ground_truth[jnt][0]

            # Assigning Bounding Box coordinates per person
            x_coord = dataset['gt'][i, 0, :, 0][np.where(dataset['gt'][i, 0, :, 0] > -1)]
            y_coord = dataset['gt'][i, 0, :, 1][np.where(dataset['gt'][i, 0, :, 1] > -1)]

            min_x = np.min(x_coord)
            max_x = np.max(x_coord)
            min_y = np.min(y_coord)
            max_y = np.max(y_coord)

            dataset['bbox_coords'][i, 0] = np.array([min_x, min_y, max_x, max_y])

            dataset['num_gt'][i, 0] = num_gt[0]

            dataset['normalizer'][i, 0] = normalizer[0]

            if self.load_images:
                dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        dataset['img'] = np.array(dataset['img'], dtype=object)
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        logging.info('LSP dataset description:')
        logging.info('Length (#images): {}'.format(dataset['gt'].shape[0]))

        return dataset


    def create_train_validate(self, dataset: dict) -> (dict, dict):
        """
        Split the dataset into train and validate dataset
        """

        # Separate train and validate
        train_idx = []
        val_idx = []
        for i in range(len(dataset['name'])):
            if dataset['split'][i] == 0:
                train_idx.append(i)
            else:
                assert dataset['split'][i] == 1, \
                    "Split has value: {}, should be either 0 or 1".format(dataset['split'][i])
                val_idx.append(i)

        train_dataset = {}
        val_dataset = {}
        for key in dataset.keys():
            if key == 'img' and (self.load_images is False):
                train_dataset[key] = dataset[key]  # Empty numpy array
                val_dataset[key] = dataset[key]
                continue

            train_dataset[key] = dataset[key][train_idx]
            val_dataset[key] = dataset[key][val_idx]

        return train_dataset, val_dataset


    def merge_dataset(self, datasets: dict, indices: int) -> dict:
        """
        Combine two or more datasets
        """

        assert type(datasets) == list and len(datasets) != 0
        assert len(datasets) == len(indices)

        for i in range(len(datasets) - 1):
            assert datasets[i].keys() == datasets[i+1].keys(), "Dataset keys do not match"

        # Merge datasets
        merged_dataset = {}
        for key in datasets[0].keys():
            if key == 'img' and (self.load_images is False):
                merged_dataset['img'] = np.array([])
                continue
            merged_dataset[key] = np.concatenate(
                [data[key][index_] for index_, data in zip(indices, datasets)], axis=0)

        # Sampling based on indices
        merged_dataset['index'] = np.arange(merged_dataset['name'].shape[0])

        return merged_dataset


    def set_augmentation(self, augment: bool) -> None:
        """
        Set augmentation flag
        """
        if augment: self.augmentation_flag = True
        else: self.augmentation_flag = False
            

    def mpii_single_person_extractor(self, mpii_dataset: dict) -> dict:
        """
        Create multiple single person images if two or more persons in image
        """

        try:
            max_persons = self.coco['max_person_in_img']
        except AttributeError:
            max_persons = self.mpii['max_person_in_img']

        dataset = {
            'img': [],
            'name': [],
            'gt': np.empty(shape=(0, max_persons, self.conf.experiment_settings['num_hm'], 3)),
            'dataset': [],
            'num_gt': np.empty(shape=(0, max_persons)),
            'split': [],
            'num_persons': np.empty(shape=(0, 1)),
            'normalizer': np.empty(shape=(0, max_persons)),
            'bbox_coords': np.empty(shape=(0, max_persons, 4))
        }


        for i in range(len(mpii_dataset['name'])):
            for p in range(int(mpii_dataset['num_persons'][i][0])):
                if self.load_images:
                    dataset['img'].append(mpii_dataset['img'][i])
                dataset['name'].append(mpii_dataset['name'][i][:-4] + '_{}.jpg'.format(p))
                dataset['dataset'].append(mpii_dataset['dataset'][i])
                dataset['split'].append(mpii_dataset['split'][i])

                gt_ = -1 * np.ones_like(mpii_dataset['gt'][i])
                gt_[0] = mpii_dataset['gt'][i, p]
                dataset['gt'] = np.concatenate(
                    [dataset['gt'], gt_.reshape(1, max_persons, self.conf.experiment_settings['num_hm'], 3)],
                    axis=0)

                num_gt_ = np.zeros_like(mpii_dataset['num_gt'][i])
                num_gt_[0] = mpii_dataset['num_gt'][i, p]
                dataset['num_gt'] = np.concatenate([dataset['num_gt'], num_gt_.reshape(1, max_persons)],
                                               axis=0)

                normalizer_ = np.zeros_like(mpii_dataset['normalizer'][i])
                normalizer_[0] = mpii_dataset['normalizer'][i, p]
                dataset['normalizer'] = np.concatenate(
                    [dataset['normalizer'], normalizer_.reshape(1, max_persons)],
                    axis=0)

                dataset['num_persons'] = np.concatenate(
                    [dataset['num_persons'], np.array([1]).reshape(1, 1)],
                    axis=0)

                bbox_ = np.zeros_like(mpii_dataset['bbox_coords'][i])
                bbox_[0] = mpii_dataset['bbox_coords'][i, p]
                dataset['bbox_coords'] = np.concatenate(
                    [dataset['bbox_coords'],bbox_.reshape(1, max_persons, 4)],
                    axis=0)

        dataset['img'] = np.array(dataset['img'], dtype=object)
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        return dataset

    def coco_single_person_extractor(self, coco_dataset: dict) -> dict:
        """
        Create multiple single person images if two or more persons in image
        """

        max_persons = self.coco['max_person_in_img']
        dataset = {
            'img': [],
            'name': [],
            'gt': np.empty(shape=(0, max_persons, len(coco_jnt_to_idx), 3)),
            'dataset': [],
            'num_gt': np.empty(shape=(0, max_persons)),
            'split': [],
            'num_persons': np.empty(shape=(0, 1)),
            'normalizer': np.empty(shape=(0, max_persons)),
            'bbox_coords': np.empty(shape=(0, max_persons, 4))
        }


        for i in tqdm(range(len(coco_dataset['name']))):
            for p in range(int(coco_dataset['num_persons'][i][0])):
                dataset['img'].append(coco_dataset['img'][i])
                dataset['name'].append("{:012d}".format(coco_dataset['name'][i]) + '_{}.jpg'.format(p))
                dataset['dataset'].append(coco_dataset['dataset'][i])
                dataset['split'].append(coco_dataset['split'][i])

                gt_ = -1 * np.ones_like(coco_dataset['gt'][i])
                gt_[0] = coco_dataset['gt'][i, p]
                dataset['gt'] = np.concatenate(
                    [dataset['gt'], gt_.reshape(1, max_persons, len(coco_jnt_to_idx), 3)],
                    axis=0)

                num_gt_ = np.zeros_like(coco_dataset['num_gt'][i])
                num_gt_[0] = coco_dataset['num_gt'][i, p]
                dataset['num_gt'] = np.concatenate([dataset['num_gt'], num_gt_.reshape(1, max_persons)],
                                               axis=0)

                normalizer_ = np.zeros_like(coco_dataset['normalizer'][i])
                normalizer_[0] = coco_dataset['normalizer'][i, p]
                dataset['normalizer'] = np.concatenate(
                    [dataset['normalizer'], normalizer_.reshape(1, max_persons)],
                    axis=0)

                dataset['num_persons'] = np.concatenate(
                    [dataset['num_persons'], np.array([1]).reshape(1, 1)],
                    axis=0)

                bbox_ = np.zeros_like(coco_dataset['bbox_coords'][i])
                bbox_[0] = coco_dataset['bbox_coords'][i, p]
                dataset['bbox_coords'] = np.concatenate(
                    [dataset['bbox_coords'],bbox_.reshape(1, max_persons, 4)],
                    axis=0)

        dataset['img'] = np.array(dataset['img'], dtype=object)
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        return dataset

    def mpii_all_joints(self, mpii_dataset: dict) -> dict:
        """
        Select images where person has all joints
        """

        # [:, 0, :, 2] --> all images, first person, all joints, visibility

        #We stop at 14 because that is the real number of joints in MPII as opposed to 17 (if use COCO)
        all_joint_indices = mpii_dataset['gt'][:, 0, :14, 2] > -0.5   # Identify indices: occluded and visible joints
        all_joint_indices = np.all(all_joint_indices, axis=1)

        for key in mpii_dataset.keys():
            if key == 'img' and (self.load_images is False):
                mpii_dataset['img'] = mpii_dataset['img']
                continue

            mpii_dataset[key] = mpii_dataset[key][all_joint_indices]

        return mpii_dataset

    def coco_all_joints(self, coco_dataset: dict) -> dict:
        """
        Select images where person has all joints
        """

        # [:, 0, :, 2] --> all images, first person, all joints, visibility
        all_joint_indices = coco_dataset['gt'][:, 0, :, 2] > -0.5   # Identify indices: occluded and visible joints
        all_joint_indices = np.all(all_joint_indices, axis=1)

        for key in coco_dataset.keys():
            if key == 'img' and (self.load_images is False):
                coco_dataset['img'] = coco_dataset['img']
                continue

            coco_dataset[key] = coco_dataset[key][all_joint_indices]

        return coco_dataset
    
    def coco_num_gt(self, coco_dataset:dict) -> dict:
        num_gt_big_enough = coco_dataset['num_gt'][:,0] > 0.5
        for key in coco_dataset.keys():
            if key == 'img' and (self.load_images is False):
                coco_dataset['img'] = coco_dataset['img']
                continue

            coco_dataset[key] = coco_dataset[key][num_gt_big_enough]
        
        return coco_dataset

    def coco_bbox_size(self, coco_dataset: dict) -> dict:
        """
        Select images where the bbox size is enough
        """
        normalizer_big_enough = coco_dataset['normalizer'][:,0] > 150
        for key in coco_dataset.keys():
            if key == 'img' and (self.load_images is False):
                coco_dataset['img'] = coco_dataset['img']
                continue

            coco_dataset[key] = coco_dataset[key][normalizer_big_enough]

        return coco_dataset
    
    def resize_image(self, image_: float, gt: float, target_size: list) -> (float, float, dict):
        '''

        :return:
        '''
        # Compute the aspect ratios
        image_aspect_ratio = image_.shape[0] / image_.shape[1]
        tgt_aspect_ratio = target_size[0] / target_size[1]

        # Compare the original and target aspect ratio
        if image_aspect_ratio > tgt_aspect_ratio:
            # If target aspect ratio is smaller, scale the first dim
            scale_factor = target_size[0] / image_.shape[0]
        else:
            # If target aspect ratio is bigger or equal, scale the second dim
            scale_factor = target_size[1] / image_.shape[1]

        # Compute the padding to fit the target size
        pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
        pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

        output_img = np.zeros(target_size, dtype=image_.dtype)

        # Write scaled size in reverse order because opencv resize
        scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

        padding_u = int(pad_u / 2)
        padding_v = int(pad_v / 2)

        im_scaled = cv2.resize(image_, scaled_size)
        # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

        output_img[padding_u : im_scaled.shape[0] + padding_u,
                   padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

        gt *= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)
        gt[:, :, 0] += padding_u
        gt[:, :, 1] += padding_v

        scale_params = {'scale_factor': scale_factor, 'padding_u': padding_u, 'padding_v': padding_v}

        return output_img, gt, scale_params


    def augmentation(self, transform: list) -> albu.Compose:
        """
        Albumentation objects for augmentation in getitem
        """
        return albu.Compose(
            transform, p=1, keypoint_params=albu.KeypointParams(format='yx', remove_invisible=False))
