import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    # Combines two sets of matches (m1 and m2) in a way that results in a final set of matches without conflicts.
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    # Converts the given indices to match pairs while filtering out matches with costs higher than the given threshold.
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    # Computes the optimal assignment between two sets using the Jonker-Volgenant
    # algorithm with the given cost matrix and threshold.
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def linear_assignment_center(cost_matrix1, cost_matrix2, thresh_iou, thresh_center):
    # Computes the optimal assignment between two sets using the Jonker-Volgenant
    # algorithm with the given cost matrices and threshold.
    if cost_matrix1.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix1.shape[0])), tuple(range(cost_matrix1.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix1, extend_cost=True, cost_limit=thresh_iou)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
            # set the cost for the matched element to a large value in cost_matrix2
            cost_matrix2[ix, :] = np.inf
            cost_matrix2[:, mx] = np.inf

    # Convert matches to a NumPy array
    matches = np.asarray(matches)
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    # Convert unmatched_a and unmatched_b to arrays of integers
    unmatched_a = np.asarray(unmatched_a).astype(int)
    unmatched_b = np.asarray(unmatched_b).astype(int)

    # extract the unmatched rows and columns from cost_matrix2
    cost_matrix2_unmatched = cost_matrix2[unmatched_a[:, None], unmatched_b]

    # use the Jonker-Volgenant algorithm with cost_matrix2_unmatched
    if cost_matrix2_unmatched.size > 0:
        _, x2, y2 = lap.lapjv(cost_matrix2_unmatched, extend_cost=True, cost_limit=thresh_center)
        for ix, mx in enumerate(x2):
            if mx >= 0:
                matches = np.vstack([matches, [unmatched_a[ix], unmatched_b[mx]]])

                print('Done with center.')

    if matches.size > 0:
        unmatched_a = [i for i in range(cost_matrix1.shape[0]) if i not in matches[:, 0]]
        unmatched_b = [i for i in range(cost_matrix1.shape[1]) if i not in matches[:, 1]]
    else:
        unmatched_a = list(range(cost_matrix1.shape[0]))
        unmatched_b = list(range(cost_matrix1.shape[1]))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)

    return matches, unmatched_a, unmatched_b


def cosine_similarity_matrix(A, B):
    A = np.array(A)
    B = np.array(B)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    similarity_matrix = np.dot(A_norm, B_norm.T)
    return similarity_matrix

def cosine_similarity_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)))

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        afeatures = atracks
        bfeatures = btracks
    else:
        afeatures = [list(track.id_feature.values())[-1] if track.id_feature else None for track in atracks]
        bfeatures = [track.init_id_feature for track in btracks]
    _cosine_similarity = cosine_similarity_matrix(afeatures, bfeatures)
    cost_matrix = 1 - _cosine_similarity

    return cost_matrix

def cosine_similarity_distance_det(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)))

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        afeatures = atracks
        bfeatures = btracks
    else:
        afeatures = [track.init_id_feature for track in atracks]
        bfeatures = [track.init_id_feature for track in btracks]
    _cosine_similarity = cosine_similarity_matrix(afeatures, bfeatures)
    cost_matrix = 1 - _cosine_similarity

    return cost_matrix

def cosine_similarity_distance_mean(atracks, btracks, num_of_frames):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)))

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        afeatures = atracks
        bfeatures = btracks
    else:
        afeatures = [track.id_feature for track in atracks]
        bfeatures = [track.init_id_feature for track in btracks]

    cost_matrix = np.empty((len(atracks), len(btracks)))

    for i, afeature_set in enumerate(afeatures):
        similarities = []
        for j, afeature in enumerate(afeature_set.values()):
            _cosine_similarity = cosine_similarity_matrix([afeature], bfeatures)
            similarities.append(_cosine_similarity)

            # num of frames
            if j == num_of_frames:
                break

        mean_similarity = np.mean(similarities, axis=0)
        cost_matrix[i] = 1 - mean_similarity

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    if (len(tracks) > 0 and isinstance(tracks[0], np.ndarray)) or (len(detections) > 0 and isinstance(detections[0], np.ndarray)):
        track_features = tracks
        det_features = detections
    else:
        track_features = [track.id_feature for track in tracks]
        det_features = [track.init_id_feature for track in detections]

    smooth_track_features = []
    for i, afeature_set in enumerate(track_features):
        smooth_track_feature = None
        for j, afeature in enumerate(afeature_set.values()):
            if smooth_track_feature is None:
                smooth_track_feature = afeature
            else:
                smooth_track_feature = 0.9*smooth_track_feature + 0.1*afeature
        smooth_track_feature /= np.linalg.norm(smooth_track_feature)
        smooth_track_features.append(smooth_track_feature)

    cost_matrix = np.maximum(0.0, cdist(smooth_track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix

# def embedding_distance(tracks, detections, metric='cosine'):
#     # Computes the distance between track and detection features using a given distance metric (default is 'cosine').
#     """
#     :param tracks: list[STrack]
#     :param detections: list[BaseTrack]
#     :param metric:
#     :return: cost_matrix np.ndarray
#     """
#
#     cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
#     if cost_matrix.size == 0:
#         return cost_matrix
#     det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
#     #for i, track in enumerate(tracks):
#         #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
#     track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
#     cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
#     return cost_matrix


def bbox_dist(axyahs1, axyahs2):
    num_axyahs1 = len(axyahs1)
    num_axyahs2 = len(axyahs2)

    # Calculate the centers of the bounding boxes
    centers1 = np.column_stack((axyahs1[:, 0], axyahs1[:, 1]))
    centers2 = np.column_stack((axyahs2[:, 0], axyahs2[:, 1]))

    # Calculate the distance matrix
    dist_matrix = np.zeros((num_axyahs1, num_axyahs2))
    for i in range(num_axyahs1):
        for j in range(num_axyahs2):
            dist_matrix[i, j] = np.sqrt(np.sum((centers1[i] - centers2[j]) ** 2))

    # Clip the distance matrix
    dist_matrix = np.clip(dist_matrix, None, 5)

    min_value = np.min(dist_matrix)
    max_value = np.max(dist_matrix)
    normalized_dists = (dist_matrix - min_value) / (max_value - min_value)

    # normalized_dists -= 1

    return normalized_dists

def center_distance_cal(axyahs, bxyahs):
    # Computes the Intersection over Union (IoU) between two sets of bounding boxes.
    """
    Compute cost based on IoU
    :type atlbrs: list[xyah] | np.ndarray
    :type atlbrs: list[xyah] | np.ndarray

    :rtype ious np.ndarray
    """
    dists = np.zeros((len(axyahs), len(bxyahs)), dtype=np.float)
    if dists.size == 0:
        return dists

    dists = bbox_dist(
        np.ascontiguousarray(axyahs, dtype=np.float),
        np.ascontiguousarray(bxyahs, dtype=np.float)
    )
    return dists

def center_distances(atracks, btracks):
    # Computes the IoU distance (1 - IoU) between two sets of tracks.
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_xyah(track.tlwh) for track in atracks]
        btlbrs = [track.tlwh_to_xyah(track.tlwh) for track in btracks]
    _center_distances = center_distance_cal(atlbrs, btlbrs)
    # cost_matrix = 1 - _center_distances

    return _center_distances

def ious(atlbrs, btlbrs):
    # Computes the Intersection over Union (IoU) between two sets of bounding boxes.
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    # Computes the IoU distance (1 - IoU) between two sets of tracks.
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def enlarge_bounding_boxes(bboxes, area_factor=1.1, img_height=1088, img_width=1920, size_threshold=None):
    enlarged_bboxes = []

    for bbox in bboxes:
        t, b, l, r = bbox
        height = b - t
        width = r - l

        if size_threshold is not None:
            # Only enlarge the bounding boxes above the size threshold
            if height * width < size_threshold:
                enlarged_bboxes.append(bbox)
                continue

        # Calculate the new width and height based on the area factor
        new_area = height * width * area_factor
        aspect_ratio = width / height

        # Update the new height and width based on the aspect ratio
        new_height = np.sqrt(new_area / aspect_ratio)
        new_width = new_height * aspect_ratio

        height_diff = (new_height - height) / 2
        width_diff = (new_width - width) / 2

        new_t = max(0, t - height_diff)
        new_b = min(img_height, b + height_diff)
        new_l = max(0, l - width_diff)
        new_r = min(img_width, r + width_diff)

        enlarged_bbox = np.array([new_t, new_b, new_l, new_r])
        enlarged_bboxes.append(enlarged_bbox)

    return enlarged_bboxes

def iou_distance_enlarge(atracks, btracks):
    # Computes the IoU distance (1 - IoU) between two sets of tracks.
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    atlbrs = enlarge_bounding_boxes(atlbrs, size_threshold=4096)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    # Computes the IoU distance (1 - IoU) between two sets of tracks, considering their predicted bounding boxes.
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix




def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    # Updates the cost matrix based on the gating distance,
    # which considers the uncertainty in the track state estimates. // 23.04.16 Inpyo
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    # Fuses the cost matrix with motion information using a specified weight (lambda_). // 23.04.16 Inpyo
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    # Fuses the cost matrix with IoU similarity. // 23.04.16 Inpyo
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    # Fuses the cost matrix with detection scores. // 23.04.16 Inpyo
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score_three(iou_cost_matrix, id_sim_matrix, detections):
    # Fuses the cost matrix with detection scores. // 23.04.16 Inpyo
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1 - iou_cost_matrix
    id_sim = 1 - id_sim_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(iou_cost_matrix.shape[0], axis=0)
    # fuse_sim = iou_sim * id_sim * det_scores
    # fuse_sim = iou_sim * det_scores
    fuse_sim = iou_sim * id_sim
    # fuse_sim = id_sim
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score_five(iou_cost_matrix, id_sim_matrix, detections, color_dist_matrix, template_sim_matrix):
    # Fuses the cost matrix with detection scores. // 23.04.16 Inpyo
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1 - iou_cost_matrix
    id_sim = 1 - id_sim_matrix
    cor_sim = 1 - color_dist_matrix
    tem_sim = 1 - template_sim_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(iou_cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * cor_sim * tem_sim
    # fuse_sim = iou_sim * id_sim * cor_sim * tem_sim
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score_six(iou_cost_matrix, id_sim_matrix, detections, color_dist_matrix, template_sim_matrix, center_dist):
    # Fuses the cost matrix with detection scores. // 23.04.16 Inpyo
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix
    iou_sim = 1 - iou_cost_matrix
    id_sim = 1 - id_sim_matrix
    cor_sim = 1 - color_dist_matrix
    tem_sim = 1 - template_sim_matrix
    center_dist = 1 - center_dist
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(iou_cost_matrix.shape[0], axis=0)
    # fuse_sim = iou_sim *  cor_sim * tem_sim * center_dist
    fuse_sim = iou_sim * id_sim * cor_sim * tem_sim * center_dist
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def edge_detection(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

def template_matching_similarity(a_roi, b_roi):
    a_roi_resized = cv2.resize(a_roi, (64, 64), interpolation=cv2.INTER_AREA)
    b_roi_resized = cv2.resize(b_roi, (64, 64), interpolation=cv2.INTER_AREA)

    mse = np.mean((a_roi_resized - b_roi_resized) ** 2)
    normalized_mse = mse / (255*255)
    return normalized_mse

def color_histogram_similarity(a_roi, b_roi):
    if a_roi is None or b_roi is None:
        return 0

    # Compute color histograms for both images
    a_hist = cv2.calcHist([a_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    b_hist = cv2.calcHist([b_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize histograms
    cv2.normalize(a_hist, a_hist)
    cv2.normalize(b_hist, b_hist)

    # Compute Bhattacharyya distance
    bhattacharyya_distance = cv2.compareHist(a_hist, b_hist, cv2.HISTCMP_BHATTACHARYYA)

    # Convert Bhattacharyya distance to similarity (1 - distance)
    similarity = 1 - bhattacharyya_distance

    return similarity

def template_matching_distance(atracks, btracks, prev_img, curr_img):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.prev_tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    num_atracks = len(atlbrs)
    num_btracks = len(btlbrs)
    cost_matrix = np.zeros((num_atracks, num_btracks), dtype=np.float)

    for i, atrack in enumerate(atracks):
        a_bbox = atrack.prev_tlbr
        a_roi = get_roi_from_bbox(a_bbox, prev_img)

        for j, btrack in enumerate(btracks):
            b_bbox = btrack.tlbr
            b_roi = get_roi_from_bbox(b_bbox, curr_img)

            if a_roi is None or b_roi is None:
                cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, j] = template_matching_similarity(a_roi, b_roi)

    return cost_matrix

def color_histogram_distance(atracks, btracks, prev_img, curr_img):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.prev_tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    num_atracks = len(atlbrs)
    num_btracks = len(btlbrs)
    cost_matrix = np.zeros((num_atracks, num_btracks), dtype=np.float)

    for i, atrack in enumerate(atracks):
        a_bbox = atrack.prev_tlbr
        a_roi = get_roi_from_bbox(a_bbox, prev_img)

        for j, btrack in enumerate(btracks):
            b_bbox = btrack.tlbr
            b_roi = get_roi_from_bbox(b_bbox, curr_img)

            if a_roi is None or b_roi is None:
                cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, j] = 1 - color_histogram_similarity(a_roi, b_roi)

    return cost_matrix

def clip_bbox(bbox, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox

    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

    return int(x1), int(y1), int(x2), int(y2)

def get_roi_from_bbox(bbox, img):
    x1, y1, x2, y2 = clip_bbox(bbox, img.shape)

    if y2 > y1 and x2 > x1:
        roi = img[y1:y2, x1:x2, :]
    elif y2 < y1 and x2 < x1:
        roi = img[y2:y1, x2:x1, :]
    elif y2 < y1 and x2 > x1:
        roi = img[y2:y1, x1:x2, :]
    elif y2 > y1 and x2 < x1:
        roi = img[y1:y2, x2:x1, :]
    else:
        roi = None

    return roi

def keep_top_n(cost_matrix, n=3):
    num_rows, num_cols = cost_matrix.shape
    modified_cost_matrix = np.full((num_rows, num_cols), 1.0, dtype=np.float)

    if num_rows == 0 or num_cols == 0:
        return modified_cost_matrix

    for col in range(num_cols):
        col_data = cost_matrix[:, col]
        finite_indices = np.isfinite(col_data)

        if np.any(finite_indices):
            col_data_finite = col_data[finite_indices]
            sorted_indices = np.argsort(col_data_finite)
            top_n_indices = sorted_indices[:min(n, len(col_data_finite))]

            finite_row_indices = np.arange(num_rows)[finite_indices]
            # modified_cost_matrix[finite_row_indices[top_n_indices], col] = 0.0
            modified_cost_matrix[finite_row_indices[top_n_indices], col] = col_data_finite[top_n_indices]

    return modified_cost_matrix

def keep_top_n_zero(cost_matrix, n=3):
    num_rows, num_cols = cost_matrix.shape
    modified_cost_matrix = np.full((num_rows, num_cols), 1.0, dtype=np.float)

    if num_rows == 0 or num_cols == 0:
        return modified_cost_matrix

    for col in range(num_cols):
        col_data = cost_matrix[:, col]
        finite_indices = np.isfinite(col_data)

        if np.any(finite_indices):
            col_data_finite = col_data[finite_indices]
            sorted_indices = np.argsort(col_data_finite)
            top_n_indices = sorted_indices[:min(n, len(col_data_finite))]

            finite_row_indices = np.arange(num_rows)[finite_indices]
            modified_cost_matrix[finite_row_indices[top_n_indices], col] = 0.0
            # modified_cost_matrix[finite_row_indices[top_n_indices], col] = col_data_finite[top_n_indices]

    return modified_cost_matrix