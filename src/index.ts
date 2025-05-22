import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { loadSummarizationChain } from "langchain/chains";
import dotenv from 'dotenv';

import fs from "fs";

dotenv.config();

// OpenAIモデル初期化
const llm = new ChatOpenAI({
  temperature: 0.1,
  model: "gpt-4.1",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const text = fs.readFileSync("/Users/rchaser53/Desktop/gpt-repository-loader/node_modules/argparse/argparse.js", "utf-8");

// テキストを文書オブジェクトに変換
// const text = "ここに要約したい長文を入れる...";
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const documents = await splitter.createDocuments([text]);

// 要約チェーンを作成
const chain = await loadSummarizationChain(llm, {
  type: "map_reduce",
});

// 要約を実行
const summary = await chain.invoke({
  input_documents: documents,
});
console.log(summary);
