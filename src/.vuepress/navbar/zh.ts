import { navbar } from "vuepress-theme-hope";

export const zhNavbar = navbar([
  "/",
  // todo update nav
  // { text: "演示", icon: "discover", link: "/demo/" },
  { text: "博文", 
    icon: "blog", 
    prefix: "/posts/" ,
  children:[{ text: "知识笔记", icon: "note", link: "notes/" },
  { text: "泉州忆往昔", icon: "like", link: "hometown/" }]},
  { text: "归档", icon: "categoryselected", link: "/timeline/"}
]);
