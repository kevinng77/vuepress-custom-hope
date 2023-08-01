<template>
    <div class="chatbot-wrapper">
        <el-container class="layout-container-demo" style="height: 500px">
            <el-container>
            <el-main>
                <el-scrollbar ref="scrollbar">
                    <MessageBox :update_scrollbar="update_scrollbar"/>
                </el-scrollbar>
            </el-main>
            <el-footer height="unset">
                <MessageInput :update_scrollbar="update_scrollbar" />
            </el-footer>
            </el-container>
        </el-container>
    </div>
</template>

<script>
import MessageInput from './MessageInput.vue';
import MessageBox from './MessageBox.vue';
export default {
    name: "ChatbotWindow",
    components:{
        MessageInput,
        MessageBox
    },
    data(){
        return {
            scroll_distance: 0
        }
    },
    methods: {
        update_scrollbar(height){
            this.$refs.scrollbar.scrollTo(0,height)
        }
    }
}
</script>




<style scoped>
.chatbot-wrapper {
    width: 66%;
    margin: 3rem auto;
    background-color: #ffffff;
    margin: 3rem auto;
    border-radius: 1rem;
    border: 1px solid var(--el-border-color);
    box-shadow: var(--el-box-shadow-light);
}
.el-footer {
    background-color: #e7e7e7;
    padding: 0.7rem 1.4rem;
    border-end-end-radius: 1rem;
    border-end-start-radius: 1rem;
}
.layout-container-demo .el-header {
  position: relative;
  background-color: var(--el-color-primary-light-7);
  color: var(--el-text-color-primary);
}
.layout-container-demo .el-aside {
  color: var(--el-text-color-primary);
  background: var(--el-color-primary-light-8);
}
.layout-container-demo .el-menu {
  border-right: none;
}
.layout-container-demo .el-main {
  padding: 0;
}
.layout-container-demo .toolbar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  right: 20px;
}
</style>