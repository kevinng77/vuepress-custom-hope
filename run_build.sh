export NODE_OPTIONS="--max-old-space-size=8192"
echo "npm version check: 9.6.7?"
echo `npm -v`
npm run docs:build
cp -r ./src/.vuepress/dist/* ~/test/dist/
cd /home/kevin/test/dist
echo `pwd`
git add .
git commit -m "add blog"
git push
git push kmtencent