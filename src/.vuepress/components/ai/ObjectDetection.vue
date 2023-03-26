<template>
  <div class="">
      <el-container style="height: 600px; border: 1px solid #eee;">
        <el-container v-loading="loading"     
              element-loading-text="记载中..."
              element-loading-background="rgba(122, 122, 122, 0.8)">
            <el-image 
              style="height: 100%; width: 100%"
              :src="dialogImageUrl" fit='contain'>
            </el-image>
          <el-footer height="120px">
            <el-row style="height: 120px" justify="center" :gutter="20" align="middle">
              <el-col v-for="fit in [0,1,2]"  :key="fit" :span="5">
                <el-image v-if="sampleimgs.length > fit" 
                 :src="sampleimgs[fit].url" fit='contain' 
                 @click="selectSampleImg(sampleimgs[fit].url)" 
                 style="cursor: pointer;height: 90px;width: 120px;background:#fdfdfd" />
              </el-col>
              <el-col :span="8">
                <div>
                <el-upload
                  ref="upload"
                  class="upload-demo"
                  :action="action"
                  :data="{task:'a task'}"
                  list-type="picture"
                  :show-file-list="false"
                  :limit="1"
                  :on-exceed="handleExceed"
                  :on-progress="handleProgress"
                  :on-success="handleSuccess"
                  :on-change="handleChange"
                  :auto-upload="false"
                  accept=".jpg,.png,.jpeg"
                >
                  <template #trigger>
                    <el-button type="primary" @click="uploadImg" >选择你的图片</el-button>
                  </template>
                  <el-button style="margin-left:0.75rem" type="success" @click="submitUpload">
                    上传并预测
                  </el-button>
                </el-upload>
                </div>
              </el-col>
            </el-row>
          </el-footer>
        </el-container>  
    </el-container>
  </div>
</template>


<script setup lang="ts">
// import {ElContainer, ElFooter, ElRow, ElCol, ElUpload, ElButton, ElImage} from 'element-plus'
// import { ElLoading } from 'element-plus/lib/components/loading'
import { genFileId } from 'element-plus'
import { ref } from 'vue'
import type { UploadInstance,  UploadProps, UploadRawFile } from 'element-plus'
import { usePageFrontmatter } from "@vuepress/client";
import axios from 'axios'

const upload = ref<UploadInstance>()
const frontmatter = usePageFrontmatter();

// 上传引擎参数
const loading = ref(false)
const img_not_predict = ref(true)  // 当前选中的图片是否被预测过了，用于调整 onChange bug
const sampleimgs = ref([])  // 样本图片
const dialogImageUrl = ref("")  // 显示当前图片的url
const is_sample_img = ref(true)  // 当前图片是否是样本图片
const action = ref("")
// 定义上传引擎参数
sampleimgs.value = frontmatter.value.sampleimgs ?? [{}, {}, {}]
action.value = frontmatter.value.action ?? ""
dialogImageUrl.value = sampleimgs.value[0].url ?? ""

// 示例图片，通过传参获取
const selectSampleImg = (url) => {
    dialogImageUrl.value = url
    is_sample_img.value = true
}

const uploadImg = () => {
    // 上传用户自定义图片
    img_not_predict.value = true
    is_sample_img.value = false
}

const handleExceed: UploadProps['onExceed'] = (files, uploadFiles) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
}

const handleChange: UploadProps['onChange'] = ( uploadFile, )=>{
  if( img_not_predict.value == true ){
    dialogImageUrl.value = uploadFile.url!
  }
}

const submitUpload = () => {
  if (is_sample_img.value){
    getSampleImageResult()
  }else{
      upload.value!.submit()
  }
}

const getSampleImageResult = () => {
    axios.post(action.value, {
      "img_url": dialogImageUrl.value,
      "is_sample_img": true
      }, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    ).then(function (response) {
      console.log(response.data)
      dialogImageUrl.value = response.data.img! ?? response.data.url
})
    .catch(function (error) {
      console.log(error);
    });
}

const handleProgress: UploadProps['onProgress'] =() =>{
  loading.value = true;
}

const handleSuccess = (response) =>{
    loading.value = false
    dialogImageUrl.value = response.img!
    img_not_predict.value = false
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>  
  .el-footer {
    background: rgb(189, 189, 189);
    z-index: 9;
  }
</style>

<style>
 .el-loading-mask {
  z-index: 8 ;
}
</style>
