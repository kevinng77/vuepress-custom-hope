import { defineUserConfig } from "vuepress";
import theme from "./theme.js";
import { getDirname, path } from "@vuepress/utils";
const __dirname = getDirname(import.meta.url);
import { searchProPlugin } from "vuepress-plugin-search-pro";

// import AutoImport from 'unplugin-auto-import/vite'
// import Components from 'unplugin-vue-components/vite'
// import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
// import ElementPlus from 'unplugin-element-plus/vite'


export default defineUserConfig({
  base: "/",
  lang: "zh-CN",
  locales: {
    "/": {
      lang: "zh-CN",
      title: "记忆笔书",
      description: "kevin 记忆笔书",
    },
  },

  head: [
    ["link", { rel: "preconnect", href: "https://fonts.googleapis.com" }],
    [
      "link",
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" },
    ],
    [
      "link",
      {
        href: "https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;700&display=swap",
        rel: "stylesheet",
      },
    ],
  ],
  plugins:[

    // ElementPlus(),
    // AutoImport({
    //   resolvers: [ElementPlusResolver()],
    // }),
    // Components({
    //   resolvers: [ElementPlusResolver()],
    // }),
    searchProPlugin({
      indexContent: true,
      customFields: [
        {
          name: "category",
          getter: (page) => page.frontmatter.category ?? [],
          formatter: {
            "/": "分类：$content",
          },
        },
        {
          name: "tag",
          getter: (page) => page.frontmatter.tag ?? [],
          formatter: {
            "/": "标签：$content",
          },
        },
      ],
    }),
  ],
  theme,
  alias: {
    // "@BlogSlide": path.resolve(__dirname, "./components/BlogSlide.vue"),
    // "@ObjectDetection": path.resolve(__dirname, "./components/ai/ObjectDetection.vue"),
    // "@TestHome": path.resolve(__dirname, "./components/TestHome.vue"),

    "@theme-hope/components/HomePage": path.resolve(
      __dirname,
      "./components/BlogHome.js"
    ),


  },
  shouldPrefetch: false,
});
