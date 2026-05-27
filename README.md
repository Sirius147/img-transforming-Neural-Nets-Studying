# NeuralNetwork Study with img nets
-------

_self project_


**Process**

> Neural Network 이론 기초 복습 with simple network 구현
   
> AutoEncoder CAE 예제코드 따라치기 및 연습

> SRCNN /(이미지 화질 개선 신경망) 기존 코드 및 논문 참고 후에 신경망 customize 진행
   
   *(pytorch로 신경망 구현 및, 데이터 전처리 과정 python PIL 라이브러리 사용하여 customize 진행*

   *신경망 최적화 (정규화 및 가중치 초기화, 학습률 조정 등)*

   *실제 현실 데이터를 통한 inference과정 설계 및 실험*

**코드예시**
# using PIL
class SRdataset(Dataset):
    def __init__(self, data_path, size_gap, sub_img_size = 33, stride = 14, upscale_factor = 3):
        super(Dataset, self).__init__()
        self.data_path = glob.glob(data_path)

        # data color channel 확인
        tmp = Image.open(self.data_path[0])
        print(f"Dataset's color mode is:{tmp.getbands()}, example size is:{tmp.size}")

        # 고해상 이미지는 filter size로 인해 크기가 작아진다.
        # 줄어드는 사이즈에 맞는 label 생성을 위한 멤버변수 -> size gap
        # 실험적으로 제시된 stride = 14, upscale factor = 3
        self.sub_img_size = sub_img_size
        self.size_gap = size_gap    
        self.stride = stride
        self.upscale_factor = upscale_factor
        self.data, self.label = [] , []
        self.transform = ToTensor()
        
        # start preprocessing
        # cropping imgs
        for path in self.data_path:
            img = Image.open(path)
            self.cropping(img, self.sub_img_size, self.size_gap, self.stride)
        
        # cropping 확인
        plt.subplot(1,3,1)
        plt.imshow(self.data[0])
        plt.title("cropped data")
        plt.subplot(1,3,2)
        plt.imshow(self.label[0])
        plt.title("cropped label")
        print(f"cropped img type: {type(self.data[0])}, width: {self.data[0].width}, height: {self.data[0].height}")

        # apply bicubic interpolation
        for idx in range(len(self.data)):
            self.data[idx] = self.bicubic_interpolation(self.data[idx],self.upscale_factor)

        plt.subplot(1,3,3)
        plt.imshow(self.data[0])
        plt.title("bicubic-I data")

        # label 정규화
        for idx in range(len(self.label)):
            self.label[idx] = np.asarray(self.label[idx], dtype=np.float32)/255.0
        
        # data transforming to Tensor
        for idx in range(len(self.data)):
            self.data[idx],self.label[idx] = self.transform(self.data[idx]), self.transform(self.label[idx])
        print(f"Now datatype is:{type(self.data[0])}")
        
    # stride를 14로 돌면서 img를 sub_img_size만큼 crop하여 train data 생성, size 변화 고려하여 data에 맞는 label 생성
    def cropping(self, img, sub_img_size, size_gap, stride):
        for i in range(0, img.height - sub_img_size + 1, stride):
            for j in range(0, img.width - sub_img_size + 1, stride):
                # sample crop으로 train sub image 생성
                cropped_temp = img.crop((j, i, j+sub_img_size, i+sub_img_size))
                self.data.append(cropped_temp)
                # size 축소에 따른 label 저장
                label_temp = img.crop((j+size_gap, i+size_gap, j+sub_img_size-size_gap,\
                                       i+sub_img_size-size_gap))
                self.label.append(label_temp)

    # upscale factor는 3으로, 먼저 upscale factor로 sub-sample해준다.(1/3으로 축소), 이후 다시 upscale factor만큼 확대해준다. 흐릿해지는 효과
    def bicubic_interpolation(self, data, scale):
        w,h = data.size
        tempw, temph = w//scale, h//scale
        temp = data.resize((tempw,temph), Image.BICUBIC)
        return temp.resize((w,h),Image.BICUBIC)
        # F.interpolate(data, scale_factor= (1/scale), mode = 'bicubic')
        # F.interpolate(data, scale_factor= scale, mode = 'bicubic')  
        # return 
    
    # data 길이 반환함수
    def __len__(self):
        return len(self.data)
    
    # indexing을 위한 overiding
    def __getitem__(self,idx):
        x,y = self.data[idx],self.label[idx]
        return x,y

![스크린샷 2024-07-13 190131](https://github.com/user-attachments/assets/d58e5eff-9195-43b8-862c-d05361f1ae0a)

    


