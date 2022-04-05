# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(cv.samples.findFile("C:\\Users\\viksr\\Desktop\\Ego\\duplicates\01N08.mp4"))
# ret, frame1 = cap.read()
# prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255
# while(1):
#     ret, frame2 = cap.read()
#     if not ret:
#         print('No frames grabbed!')
#         break
#     next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
#     flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
#     bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     cv.imshow('frame2', bgr)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     elif k == ord('s'):
#         cv.imwrite('opticalfb.png', frame2)
#         cv.imwrite('opticalhsv.png', bgr)
#     prvs = next
# cv.destroyAllWindows()

# import cv2, os, torch
# import torchvision
# import numpy as np

# files = os.listdir(".\Exo")
# for file in files:
#     filename = file.split('.')[0]+".jpg"
#     count = 0
#     cap = cv2.VideoCapture(os.path.join(".\Exo", file))
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret:
#             if count%400 == 0:
#                 cv2.imwrite(os.path.join('.\exo_images', filename), frame)
#             count += 1
#         else:
# #             break
# filename = "C:\\Users\\viksr\\Desktop\\Ego\\CharadesEgo\\00V9V.mp4"
# out_filename = "C:\\Users\\viksr\\Desktop\\Ego\\CharadesEgo\\00V9V.mp4"

# from torchvision.io import read_video
# import torch
# from torchvision.models.optical_flow import raft_large
# import torchvision.transforms.functional as F
# import torchvision.transforms as T
# from torchvision.utils import flow_to_image
# # import numpy as np
# # import matplotlib.pyplot as plt

# def preprocess(batch):
#     transforms = T.Compose(
#         [
#             T.ConvertImageDtype(torch.float32),
#             T.Normalize(mean=0.5, std=0.5),
#             T.Resize(size=(520, 960)),
#         ]
#     )
#     batch = transforms(batch)
#     return batch

# plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


# def plot(imgs, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             img = F.to_pil_image(img.to("cpu"))
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     plt.tight_layout()

# video_path = "C:\\Users\\viksr\\Desktop\\Ego\\CharadesEgo\\0CCESEGO.mp4"
# # video_path = "C:\\Users\\viksr\\AppData\\Local\\Temp\\tmpx3_57vip\\basketball.mp4"
# frames, _, _ = read_video(video_path)
# frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

# img1_batch = torch.stack([frames[0], frames[1]])
# img2_batch = torch.stack([frames[10], frames[11]])

# device = "cuda" if torch.cuda.is_available() else "cpu"
# img1_batch = preprocess(img1_batch).to(device)
# img2_batch = preprocess(img2_batch).to(device)

# print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

# # If you can, run this example on a GPU, it will be a lot faster.
# model = raft_large(pretrained=".\\raft_large_C_T_SKHT_V2-ff5fadd5.pth").to(device)
# model = model.eval()

# list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
# predicted_flows = list_of_flows[-1]
# print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
# flow_imgs = flow_to_image(predicted_flows)

# im1 = F.to_pil_image(flow_imgs[0].to("cpu"))
# im1.save('1_ego.jpg')
# im2 = F.to_pil_image(flow_imgs[1].to("cpu"))
# im2.save('2_ego.jpg')