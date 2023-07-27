import { defineClientConfig } from "@vuepress/client";
import MenuLayout from "./layouts/MenuLayout";
import SearchLayout from "./layouts/SearchLayout";
import 'element-plus/dist/index.css'
// import ObjectDetection from "./components/ai/ObjectDetection.vue";
import ElementPlus from 'element-plus'

export default defineClientConfig({
  layouts: {
    // 比如这里我们将 vuepress-theme-hope 的默认布局改为自己主题下的 layouts/Layout.vue
    MenuLayout,
    SearchLayout
  },
  enhance: ({ app, router, siteData }) => {
    // app.component("ObjectDetection", ObjectDetection);

    app.use(ElementPlus);
;
  },
});