import { sidebar } from "vuepress-theme-hope";

export const zhSidebar = sidebar({
  "/": [
    "",
    {
      text: "如何使用",
      icon: "creative",
      prefix: "demo/",
      link: "demo/",
      children: "structure",
    },
    {
      text: "文章目录",
      icon: "note",
      prefix: "posts/",
      children: "structure",
    },
    {
      text: "int",
      icon: "intro",
      link: "intro/"
    },

  ]
  ,
  "/posts/hometown/":[
    {
      text: "文章",
      icon: "note",
      prefix: "articles/",
      children: "structure",
    }
  ],
  "/posts/notes/": [
    {
      text: "文章",
      icon: "note",
      prefix: "articles/",
      children: "structure",
    }
  ],
  "/posts/hometown/articles":  "heading",
  "/posts/notes/articles":  "heading"
});
