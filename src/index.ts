import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadSummarizationChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";
import dotenv from 'dotenv';
import fs from "fs";

dotenv.config();

// コマンドライン引数の解析
const args = process.argv.slice(2);
const summaryLengthArg = args.find(arg => arg.startsWith('--length='));
const filePathArg = args.find(arg => arg.startsWith('--file='));

// デフォルト値の設定
const summaryLength = summaryLengthArg ? summaryLengthArg.split('=')[1] : 'medium';
const filePath = filePathArg ? filePathArg.split('=')[1] : "/Users/rchaser53/Desktop/gpt-repository-loader/node_modules/argparse/argparse.js";

// 要約の長さに応じた指示を設定
const getLengthInstruction = (length: string): string => {
  switch (length.toLowerCase()) {
    case 'short':
    case 'brief':
      return '簡潔に2-3文で要約してください。';
    case 'medium':
    case 'normal':
      return '適度な長さ（5-8文程度）で要約してください。';
    case 'long':
    case 'detailed':
      return '詳細に10-15文程度で要約してください。';
    default:
      // 数値が指定された場合（例：--length=100）
      if (!isNaN(Number(length))) {
        return `約${length}文字程度で要約してください。`;
      }
      return '適度な長さで要約してください。';
  }
};

// OpenAIモデル初期化
const llm = new ChatOpenAI({
  temperature: 0.1,
  model: "gpt-4o-mini",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// ファイルの存在確認
if (!fs.existsSync(filePath)) {
  console.error(`エラー: ファイルが見つかりません: ${filePath}`);
  process.exit(1);
}

const text = fs.readFileSync(filePath, "utf-8");

// テキストを文書オブジェクトに変換
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 90000, // https://github.com/langchain-ai/langchainjs/issues/6854#issuecomment-2366862510
  chunkOverlap: 200,
});
const documents = await splitter.createDocuments([text]);

// カスタムプロンプトテンプレートを作成
const mapPrompt = PromptTemplate.fromTemplate(`
以下のテキストを読んで、${getLengthInstruction(summaryLength)}
重要なポイントを漏らさないように注意してください。

テキスト: {text}

要約:`);

const combinePrompt = PromptTemplate.fromTemplate(`
以下の要約を統合して、${getLengthInstruction(summaryLength)}
全体的な内容を包括的にまとめてください。

要約リスト:
{text}

最終要約:`);

// 要約チェーンを作成
const chain = await loadSummarizationChain(llm, {
  type: "map_reduce",
  combineMapPrompt: mapPrompt,
  combinePrompt: combinePrompt,
});

console.log(`ファイル: ${filePath}`);
console.log(`要約の長さ: ${summaryLength}`);
console.log('要約を実行中...\n');

// 要約を実行
const summary = await chain.invoke({
  input_documents: documents,
});

console.log('=== 要約結果 ===');
console.log(summary.text);
console.log('\n使用方法:');
console.log('node dist/index.js --file=<ファイルパス> --length=<short|medium|long|数値>');
console.log('例: node dist/index.js --file=./sample.txt --length=short');
