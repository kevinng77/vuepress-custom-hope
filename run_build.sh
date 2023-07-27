pnpm run docs:build
cp -r ./src/.vuepress/dist/* ~/test/dist/
cd /home/kevin/test/dist
echo `pwd`
git add .
git commit -m "add blog"
git push
git push kmserver