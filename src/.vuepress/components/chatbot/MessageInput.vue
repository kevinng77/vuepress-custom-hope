<template>
    <el-input
        class="input-box"
        v-model="textarea"
        :disabled="loading"
        maxlength="1000"
        placeholder="Please input"
        show-word-limit
        type="textarea"
        :rows="2"
        @keypress.enter="handleSent"
    />
</template>

<script language="Javascript1.9">
import { Promotion } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import {mapActions} from 'vuex'

export default {
    name: "MessageInput",
    data(){
        return {
            textarea: "",
            loading: false
        }
    },
    methods:{
        ...mapActions([
            'appendMessage',
            'updateResponse'
        ]),
        // https://www.builder.io/blog/stream-ai-javascript
        async OpenAIStream(prompt) {
            const res = await fetch(`http://20.2.68.156:5001/v1/chat/completions`, {
                headers: {
                'Content-Type': 'application/json',
                },
                method: 'POST',
                    body: JSON.stringify({
                    model: 'llama-2',
                    messages:[
                        {"role": "user", "content": prompt}
                    ],
                    temperature: 0,
                    stream: true,
                    max_tokens: 1000
                    })
            });

            if (res.status !== 200) {
                throw new Error('OpenAI API returned an error');
            }

            const data = res.body;

            if (!data) {
            return;
            }

            const reader = data.getReader();
            const decoder = new TextDecoder();
            let done = false;

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value);
                const lines = chunkValue.split(/\r?\n/);
                // console.log(lines)
                const parsedLines = lines
                .filter(line => line.trim() !== '')
                .map((line) => line.replace(/^data: /, "").trim())
                    .filter(line=>line !== "[DONE]") 

                for (const parsedLine of parsedLines) {
                    try {
                        let dict = JSON.parse(parsedLine);
                        let next_token = dict.choices[0].delta.content;
                        if (next_token){
                            for(let char of next_token){
                                this.updateResponse(char)
                            }
                        }
                        }
                    catch(error){
                        console.log(error)
                    }
                    }
                }
                console.log(this.$store.state.messages)
            this.loading = false
        },
        handleSent(){
            this.loading=true;
            console.log(this.loading)
            if (!this.textarea){
                ElMessage({
                showClose: true,
                message: 'Please enter a message',
                center: true,
                type: 'error',
                })
                return;
            }
            this.appendMessage({'user':'human','content':this.textarea});
            this.appendMessage({'user':'bot', 'content': ''})
            this.OpenAIStream(this.textarea);
            this.textarea = "";
        }
    },
    setup(){
        return {
            Promotion
        }
    }
}
</script>
