<template>
    <el-container class="chat-container">
        <el-main>
            <el-scrollbar ref="scrollbar" max-height="500px">
                <MessageBox :update_scrollbar="update_scrollbar"/>
            </el-scrollbar>
        </el-main>
        <el-footer height="unset">
            <MessageInput :update_scrollbar="update_scrollbar" />
        </el-footer>
    </el-container>
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
.chat-container {
    height: 600px;
}
.el-footer {
    padding: 1rem 2rem;
}
</style>