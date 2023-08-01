<template>
    <div class="box-wrapper" ref="box">
        <ul class="message-styles-box">
            <el-row v-for="message in messages" :key="message.content" 
                :class="`${message.user==='human'?'human-row':'bot-row'}`"
                :gutter="20"
                >
                <el-col :span="2" class="message-avatar">
                    <el-avatar shape="square" :size="46"  />
                </el-col>
                <el-col :span="16">
                    <el-card 
                        :class="`message-card ${message.user==='bot'?'bot-message':'human-message'}`" 
                        shadow="hover"
                        body-style="padding: .3rem 1rem; border-radius: 5rem;" 
                        >
                      <MessageMarkdown :raw="message.content" />  
                    </el-card>
                </el-col>
            </el-row>
        </ul>
    </div>
</template>

<script>
import MessageMarkdown from "./MessageMarkdown.vue"
export default {
    name: "MessageBox",
    components: {
        MessageMarkdown
    },
    data(){
        return {
            old_height: 0
        }
    },
    props: {'update_scrollbar':{
        type: Function
    }}
    ,
    computed: {
        messages(){
            return this.$store.state.messages;
        }
    },
    updated(){
        if (this.old_height !== this.$refs.box.clientHeight){ 
            this.update_scrollbar(this.$refs.box.clientHeight);
            this.old_height = this.$refs.box.clientHeight;
        }
    }
}
</script>

<style>
.message-styles-box {
    width: 95%;
    padding-inline-start: 2.5%;
    padding-inline-end: 2.5%;
}
.message-card {
    margin: 0.5rem 0;
}
.message-avatar {
    margin: 0.5rem 0;
    text-align: center;
}
.bot-message {
    background-color: rgb(231 235 236);
}
.human-row {
    flex-direction: row-reverse
}

</style>