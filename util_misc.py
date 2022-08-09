from torch import no_grad as torch_no_grad, load as torch_load


def load_gbnet_fcs_weights(model, fpath, state_dict=None):
    pass


def load_gbnet_rpn_weights(model, fpath, state_dict=None):
    pass


def load_gbnet_vgg_weights(model, fpath, state_dict=None):
    if state_dict is None:
        state_dict = torch_load(fpath)['state_dict']
    with torch_no_grad():
        model.backbone.body.conv_body[0].weight.copy_(state_dict['features.0.weight'])
        model.backbone.body.conv_body[0].bias.copy_(state_dict['features.0.bias'])
        model.backbone.body.conv_body[2].weight.copy_(state_dict['features.2.weight'])
        model.backbone.body.conv_body[2].bias.copy_(state_dict['features.2.bias'])
        model.backbone.body.conv_body[5].weight.copy_(state_dict['features.5.weight'])
        model.backbone.body.conv_body[5].bias.copy_(state_dict['features.5.bias'])
        model.backbone.body.conv_body[7].weight.copy_(state_dict['features.7.weight'])
        model.backbone.body.conv_body[7].bias.copy_(state_dict['features.7.bias'])
        model.backbone.body.conv_body[10].weight.copy_(state_dict['features.10.weight'])
        model.backbone.body.conv_body[10].bias.copy_(state_dict['features.10.bias'])
        model.backbone.body.conv_body[12].weight.copy_(state_dict['features.12.weight'])
        model.backbone.body.conv_body[12].bias.copy_(state_dict['features.12.bias'])
        model.backbone.body.conv_body[14].weight.copy_(state_dict['features.14.weight'])
        model.backbone.body.conv_body[14].bias.copy_(state_dict['features.14.bias'])
        model.backbone.body.conv_body[17].weight.copy_(state_dict['features.17.weight'])
        model.backbone.body.conv_body[17].bias.copy_(state_dict['features.17.bias'])
        model.backbone.body.conv_body[19].weight.copy_(state_dict['features.19.weight'])
        model.backbone.body.conv_body[19].bias.copy_(state_dict['features.19.bias'])
        model.backbone.body.conv_body[21].weight.copy_(state_dict['features.21.weight'])
        model.backbone.body.conv_body[21].bias.copy_(state_dict['features.21.bias'])
        model.backbone.body.conv_body[24].weight.copy_(state_dict['features.24.weight'])
        model.backbone.body.conv_body[24].bias.copy_(state_dict['features.24.bias'])
        model.backbone.body.conv_body[26].weight.copy_(state_dict['features.26.weight'])
        model.backbone.body.conv_body[26].bias.copy_(state_dict['features.26.bias'])
        model.backbone.body.conv_body[28].weight.copy_(state_dict['features.28.weight'])
        model.backbone.body.conv_body[28].bias.copy_(state_dict['features.28.bias'])
    print('load_gbnet_vgg_weights: loaded weights')
    return state_dict
 # 'roi_fmap.0.weight',
 # 'roi_fmap.0.bias',
 # 'roi_fmap.3.weight',
 # 'roi_fmap.3.bias',
 # 'score_fc.weight',
 # 'score_fc.bias',
 # 'bbox_fc.weight',
 # 'bbox_fc.bias',
 # 'rpn_head.anchors',
 # 'rpn_head.conv.0.weight',
 # 'rpn_head.conv.0.bias',
 # 'rpn_head.conv.2.weight',
 # 'rpn_head.conv.2.bias']
