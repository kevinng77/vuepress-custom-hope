<template>
      <el-carousel class="slide" :height="data.slideHeight" :interval="5000">
        <el-carousel-item  v-for="item in data.slides" :key="item">
          <el-image class="slide-img" fit='cover' 
          style="width: 100%; height: 100%" :src="item.url"  alt="" />
          <a class="slide-content slide-name" :href="item.link">{{item.name}}</a>
          <p class="slide-content" v-for="desc in item.desc" :key="desc">{{ desc }}</p>
        </el-carousel-item>
      </el-carousel>
</template>

<script>
// import { ElCarousel, ElCarouselItem, ElImage } from 'element-plus'

export default ({
  // components: {
  //   ElCarousel, ElCarouselItem, ElImage 
  // },
  computed: {
    data () {
      return this.$page.frontmatter
    },
  }

})

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