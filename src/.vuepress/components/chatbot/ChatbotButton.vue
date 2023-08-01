<template>
    <div class="chatbot-button">
        <el-button type="primary" :icon="ChatRound" 
        @click="triggerBot" size='large' circle />
        <Teleport to="body">
            <Transition  name="slide-fade">
                <div v-if="open" class="chat-window">
                    <ChatbotWindowVue />
                </div>
            </Transition>
        </Teleport>

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
    z-index: 99999999999;
}

.chat-window {
    width: 100%;
    z-index: 99999999999;
    top: 0;
    position: fixed;
}

.slide-fade-enter-active {
  transition: all 0.1s ease-out;
}

.slide-fade-leave-active {
  transition: all 0.1s cubic-bezier(1, 0.5, 0.8, 1);
}

.slide-fade-enter-from,
.slide-fade-leave-to {
  transform: translateX(20px);
  opacity: 0;
}

</style>