import { defineClientConfig } from "@vuepress/client";
import MenuLayout from "./layouts/MenuLayout";
import SearchLayout from "./layouts/SearchLayout";
import 'element-plus/dist/index.css'
import ChatbotButton from "./components/chatbot/ChatbotButton.vue"
import store from './store/index.js';
import ElementPlus from 'element-plus'

export default defineClientConfig({
  layouts: {
    // 比如这里我们将 vuepress-theme-hope 的默认布局改为自己主题下的 layouts/Layout.vue
    MenuLayout,
    SearchLayout
  },
  enhance: ({ app, router, siteData }) => {
    app.use(ElementPlus);
    app.use(store);
;
  },
  rootComponents: [ChatbotButton]
});