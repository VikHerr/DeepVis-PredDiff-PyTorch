import torch 



def max_probability_score_inverted(probabilities, number_of_model_classes, device="cuda"):
    # scores are inverted, such that: 0=ID, 1=OOD
    ood_score = 1 - probabilities.max(dim=1)[0]
    if ood_score.ndim < 2:
        return ood_score.unsqueeze(0)
    else: 
        return ood_score


def entropic_score_normalized_inverted(probabilities, number_of_model_classes, device="cuda"):
    # scores are inverted, such that: 0=ID, 1=OOD
    log_num_classes = torch.log(torch.Tensor([number_of_model_classes]))
    log_num_classes = log_num_classes.cuda() if device != "cpu" else log_num_classes
    ood_score = -(probabilities * torch.log(probabilities)).sum(dim=1) / log_num_classes
    return ood_score.unsqueeze(0)



# def infer_with_dataloader(imgdir, dataset, device, model, score_type, number_of_model_classes):
#     test_loader = loader_for_xai.get_inference_loader(imgdir, dataset, return_img_name=True, batch_size=4)
#     for data in test_loader:

#         img_batch, label_batch, img_names = data
#         if device == "cuda":
#             img_batch.cuda(), label_batch.cuda()
#         outputs = model(img_batch)
#         probabilities = torch.nn.Softmax(dim=1)(outputs)

#         if score_type == "MPS":  # the maximum probability score
#             ood_score_batch = max_probability_score_inverted(probabilities)
#         elif score_type == "ES":  # the negative entropy score (now positive + normalized)
#             ood_score_batch = entropic_score_normalized_inverted(probabilities, number_of_model_classes, device=device)

#         print("image names", img_names)
#         print("labels:", label_batch)
#         print("probabilities:", probabilities)
#         print("ood_score_batch:", ood_score_batch)


# def infer_with_dataset(imgdir, dataset, device, model, score_type, number_of_model_classes):
#     xai_selection = loader_for_xai.get_dataset(imgdir, dataset)
#     for i in range(len(xai_selection)):
#         # load one image with label
#         data = xai_selection[i]
#         img, label = data
#         label_name = xai_selection.get_class_to_idx_inverted()[label] if label >= 0 else "OOD"
#         img_batch = img[None, :]
#         label_batch = torch.Tensor([label])[None, :]

#         print(xai_selection.img_names[i])

#         if device == "cuda":
#             img_batch.cuda(), label_batch.cuda()
#         outputs = model(img_batch)
#         probabilities = torch.nn.Softmax(dim=1)(outputs)

#         if score_type == "MPS":  # the maximum probability score
#             ood_score_batch = max_probability_score_inverted(probabilities)
#         elif score_type == "ES":  # the negative entropy score (now positive + normalized)
#             ood_score_batch = entropic_score_normalized_inverted(probabilities, number_of_model_classes, device=device)

#         print("labels:", label_batch)
#         print("label_name:", label_name)
#         print("probabilities:", probabilities)
#         print("ood_score_batch:", ood_score_batch)
#         print()

