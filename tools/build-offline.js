// 把 birthday-cake.html 打成一个"断网也能开"的单文件版本。
//
// 为什么要这个：原版从 cdnjs 取 three.js、从 Google Fonts 取字体。
// 这两个域名在中国大陆基本打不开 —— 字体挂了只是字变样，three.js 挂了
// 整个画面直接出不来。凡是打算把链接发给国内的人，就该用这个版本。
//
//     node tools/build-offline.js
//
// 产出 birthday-cake-offline.html：除了文件本身，不请求任何东西。
// 双击就能看，也可以扔到任何静态托管上。

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SRC = path.join(ROOT, 'birthday-cake.html');
const OUT = path.join(ROOT, 'birthday-cake-offline.html');
const THREE_SRC = path.join(__dirname, 'three.min.js');

if (!fs.existsSync(THREE_SRC)) {
  console.error('缺少 tools/three.min.js —— 先下载：');
  console.error('  curl -sL -o tools/three.min.js \\');
  console.error('    https://cdnjs.cloudflare.com/ajax/libs/three.js/0.147.0/three.min.js');
  process.exit(1);
}

let html = fs.readFileSync(SRC, 'utf8');
const three = fs.readFileSync(THREE_SRC, 'utf8');

// 内联的脚本里只要出现 </script 就会提前把标签闭合掉，整页报废
if (/<\/script/i.test(three)) {
  console.error('three.min.js 里含 </script，不能直接内联');
  process.exit(1);
}

const before = html.length;

// ① 去掉 Google Fonts 的三个标签。字体栈里本来就有系统字体的兜底，
//    拿掉只是少一层，不会退化到无衬线默认值。
html = html
  .replace(/<link rel="preconnect" href="https:\/\/fonts\.googleapis\.com">\n?/, '')
  .replace(/<link rel="preconnect" href="https:\/\/fonts\.gstatic\.com" crossorigin>\n?/, '')
  .replace(/<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com\/[^"]*">\n?/, '');

// ② three.js 换成内联
const TAG = /<script src="https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/three\.js\/0\.147\.0\/three\.min\.js"><\/script>/;
if (!TAG.test(html)) {
  console.error('找不到 three.js 的 script 标签，页面结构可能改了');
  process.exit(1);
}
html = html.replace(TAG, '<script>\n' + three + '\n</script>');

// ③ 补上原文件指望外层骨架提供、但独立发布时没人给的三件东西。
//
//    原文件是按 Artifact 规范写的：那个骨架会替你补 doctype、charset 和
//    viewport meta。扔到 GitHub Pages 或者直接双击打开时没有骨架，三样都得自己来。
//
//    viewport 这条最要命：少了它，手机按 980px 的布局视口渲染再整体缩小，
//    画面糊成一团、问候语小得看不清 —— 而链接主要就是发给人用手机点的。
//    viewport-fit=cover 是给页面里那两处 env(safe-area-inset-*) 用的，
//    不带它，刘海屏上的内边距恒为 0。
html = '<!doctype html>\n'
     + '<meta charset="utf-8">\n'
     + '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
     + html;

// ④ 自检：不该再有任何外部请求
const left = html.match(/https?:\/\/[^"' )]+/g) || [];
const bad = left.filter(u => !/^https?:\/\/(www\.)?(w3\.org|github\.com|creativecommons)/.test(u));
if (bad.length) {
  console.error('还残留外部引用：');
  bad.forEach(u => console.error('  ' + u));
  process.exit(1);
}

// ⑤ 自检：独立打开所必需的三个头，缺一个都不算打包成功
for (const [what, re] of [
  ['doctype',      /^<!doctype html>/i],
  ['charset',      /<meta charset="utf-8">/i],
  ['viewport',     /<meta name="viewport"[^>]*width=device-width/i],
  ['viewport-fit', /<meta name="viewport"[^>]*viewport-fit=cover/i],
]) {
  if (!re.test(html)) {
    console.error('打包结果缺少 ' + what + '，独立打开会出问题');
    process.exit(1);
  }
}

fs.writeFileSync(OUT, html);
console.log('已生成 ' + path.relative(ROOT, OUT));
console.log('  ' + (before / 1024).toFixed(1) + ' KB  ->  ' + (html.length / 1024).toFixed(1) + ' KB');
console.log('  外部请求：0');
