import torch
import torch.nn as nn

class VideoFeatureMap2DConvPooler(nn.Module):

    def __init__(
        self,
        embed_dims,
    ):

        super().__init__()

        self.embed_dims = embed_dims
        self.conv2d_layer0_2x4_s24 = nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(2,4), stride=(2,4), groups=self.embed_dims) 
        self.conv2d_layer1_3x3_s11 = nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(3,3), stride=(1,1), groups=self.embed_dims)
        self.conv2d_layer2_3x3_s11 = nn.Conv2d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(3,3), stride=(1,1), groups=self.embed_dims)
        self.activation = nn.GELU()

    def forward(self, feature_map_adapted):
        """
        feature_map_adapted: Tensor (b, t, hp, wp, d)
        """

        b,t,hp,wp,d = feature_map_adapted.shape

        # Reshape / Permute for conv
        feature_map_adapted = feature_map_adapted.permute(0,1,4,2,3) # (b,t,d,hp,wp)
        feature_map_adapted= feature_map_adapted.reshape(b*t, d, hp, wp) #(b*t,d,hp,wp)

        # Apply 3D conv layers with none-linearity in between
        feature_map_2dconv_pooled = self.conv2d_layer0_2x4_s24(feature_map_adapted) # (b*t, d, 5, 5)
        feature_map_2dconv_pooled = self.activation(feature_map_2dconv_pooled)
        feature_map_2dconv_pooled = self.conv2d_layer1_3x3_s11(feature_map_2dconv_pooled) # (b*t, d, 2, 3, 3)
        feature_map_2dconv_pooled = self.activation(feature_map_2dconv_pooled)
        feature_map_2dconv_pooled = self.conv2d_layer2_3x3_s11(feature_map_2dconv_pooled) # (b*t, d, 1, 1)

        feature_map_2dconv_pooled = feature_map_2dconv_pooled.squeeze()
        feature_map_2dconv_pooled = feature_map_2dconv_pooled.reshape(b,t,d)

        feature_map_2dconv_pooled_mean = feature_map_2dconv_pooled.mean(dim=1)[None,:,:]
        
        return feature_map_2dconv_pooled , feature_map_2dconv_pooled_mean  
    


class VideoFeatureMap3DConvPooler(nn.Module):

    def __init__(
        self,
        embed_dims,
    ):

        super().__init__()

        self.embed_dims = embed_dims
        self.conv3d_layer0_2x2x2_s224 = nn.Conv3d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(2,2,2), stride=(2,2,4), groups=self.embed_dims)
        self.conv3d_layer1_2x3x3_s211 = nn.Conv3d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(2,3,3), stride=(2,1,1), groups=self.embed_dims)
        self.conv3d_layer2_2x3x3_s111 = nn.Conv3d(in_channels=self.embed_dims, out_channels=self.embed_dims, kernel_size=(2,3,3), stride=(1,1,1), groups=self.embed_dims)
        self.activation = nn.GELU()


    def forward(self, feature_map_adapted):
        """
        feature_map_adapted: Tensor (b, t, hp, hw, d)
        """

        # Reshape / Permute for conv
        feature_map_adapted = feature_map_adapted.permute(0,4,1,2,3) # (b,d,t,hp,hw)

        # Apply 3D conv layers with none-linearity in between
        feature_map_3dconv_pooled = self.conv3d_layer0_2x2x2_s224(feature_map_adapted) # (b, d, 4, 5, 5)
        feature_map_3dconv_pooled = self.activation(feature_map_3dconv_pooled)
        feature_map_3dconv_pooled = self.conv3d_layer1_2x3x3_s211(feature_map_3dconv_pooled) # (b, d, 2, 3, 3)
        feature_map_3dconv_pooled = self.activation(feature_map_3dconv_pooled)
        feature_map_3dconv_pooled = self.conv3d_layer1_2x3x3_s211(feature_map_3dconv_pooled) # (b, d, 1, 1, 1)

        feature_map_3dconv_pooled = feature_map_3dconv_pooled.squeeze()[None,:,:]

        return feature_map_3dconv_pooled   