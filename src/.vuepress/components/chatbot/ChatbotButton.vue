<template>
    <div class="chatbot-button">
        <el-button type="primary" :icon="ChatRound" 
            @click="triggerBot" size='large' circle />
        <el-dialog v-model="open" width="66%" :append-to-body=true top=2rem
         :show-close=false draggable>
            <ChatbotWindowVue />
        </el-dialog>    
    </div>
</template>


<script>
import {ChatRound} from '@element-plus/icons-vue'
import ChatbotWindowVue from './ChatbotWindow.vue'

export default {
    name: "ChatbotButton",
    components:{
        ChatbotWindowVue
    },
    data(){
        return {
            open: false
        }
    },
    methods:{
        triggerBot(){
            this.open = !this.open
        }
    },
    setup(){
        return {
            ChatRound
        }
    }
}
</script>

<style>
.chatbot-button {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 999;
}

.el-dialog {
    border-radius: 1rem;
}

.el-dialog__header {
    padding: 5px;
}

.el-dialog__body {
    padding: 0 0 1rem 0;
}

</style>