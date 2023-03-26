<template>
      <main class="page" id="main-content">
          <BreadCrumb />
          <!-- <PageTitle /> -->
           
          <main class="menu-page">
            <el-input
              v-model="input"
              placeholder="Please input"
              class="input-with-select"
            >
              <template #prepend>
                <el-select v-model="select" placeholder="Select" style="width: 115px">
                  <el-option label="Restaurant" value="1" />
                  <el-option label="Order No." value="2" />
                  <el-option label="Tel" value="3" />
                </el-select>
              </template>
              <template #append>
                <el-button type="primary" @click="searchArticle">Search</el-button>
              </template>
              </el-input>
              <div>
                {{input}}
                {{select}}
              </div>
          </main>
          <!-- <PageMeta /> -->
      </main>
</template>

<script setup lang="ts">
import { ref } from 'vue'
// import { ElCarousel, ElCarouselItem, ElImage } from 'element-plus'
import BreadCrumb from "@theme-hope/components/BreadCrumb"
import PageTitle from "@theme-hope/components/PageTitle"
import PageMeta from "@theme-hope/modules/info/components/PageMeta"
import axios from 'axios'
import { usePageFrontmatter } from "@vuepress/client";


const select = ref('')
const input = ref('')
const frontmatter = usePageFrontmatter();
const databaseurl = ref('')
const articles = ref('')


databaseurl.value = frontmatter.value.databaseurl ?? ""



const searchArticle = () => {
    axios.post(databaseurl.value, {
      "input": input.value,
      }, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    ).then(function (response) {

      dialogImageUrl.value = response.data.img! ?? response.data.url


})
    .catch(function (error) {
      console.log(error);
    });
}



</script>


<style scoped>
@font-face {
    font-family: "fangzheng";
    src: url('../public/assets/fonts/FZQKBYSJW.subset.TTF') format('truetype');
}
.slide-content {
  margin: 0.1rem auto;
  color: #fff;
  z-index: 999;
  font-size: 20px;
  font-family: fangzheng, "Noto Serif SC";
}

.slide {
  margin-bottom: 50px;
}

.slide-name {
  font-size: 40px;
  margin-bottom: 20px;
}

.slide-img {
  position: fixed;
  -webkit-filter: brightness(50%);
  filter: brightness(50%);
}

.el-carousel__item {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

</style>