<template>
    <el-text v-html="parsed_md">
    </el-text>
</template>

<script language="Javascript1.9">
import { marked } from 'marked';
import {markedHighlight} from "marked-highlight";
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.css'
marked.use(markedHighlight({
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  }
}));


export default {
    name: "MessageMarkdown",
    props: ['raw'],
    data(){
        return {
            parsed_md: marked.parse(this.raw)
        }
    },
    updated(){

    }
}
</script>

<style>

pre code.hljs {
    border-radius: .7rem;
    background-color: rgba(162, 162, 162, 0.12)
}
</style>