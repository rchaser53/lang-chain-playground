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

// テキストから特殊トークンを除去する関数
const cleanText = (text: string): string => {
  return text
    .replace(/<\|endoftext\|>/g, '') // <|endoftext|>トークンを除去
    .replace(/<\|startoftext\|>/g, '') // <|startoftext|>トークンを除去
    .replace(/<\|[^|]*\|>/g, '') // その他の特殊トークンを除去
    .replace(/\x00/g, '') // NULL文字を除去
    .trim();
};

// 進行度表示クラス
class ProgressLogger {
  private startTime: number;
  private currentStep: number = 0;
  private totalSteps: number;

  constructor(totalSteps: number) {
    this.startTime = Date.now();
    this.totalSteps = totalSteps;
  }

  logStep(stepName: string): void {
    this.currentStep++;
    const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
    const progress = Math.floor((this.currentStep / this.totalSteps) * 100);
    console.log(`[${this.currentStep}/${this.totalSteps}] (${progress}%) ${stepName} - 経過時間: ${elapsed}秒`);
  }

  logComplete(): void {
    const totalTime = Math.floor((Date.now() - this.startTime) / 1000);
    console.log(`\n✅ 処理完了 - 総実行時間: ${totalTime}秒\n`);
  }
}

// トークン制限対応のレート制限クラス
class TokenRateLimiter {
  private requests: number[] = [];
  private tokenUsage: { time: number; tokens: number }[] = [];
  private maxRequests: number;
  private maxTokens: number;
  private timeWindow: number;

  constructor(maxRequests: number = 200, maxTokens: number = 150000, timeWindow: number = 60000) {
    this.maxRequests = maxRequests;
    this.maxTokens = maxTokens;
    this.timeWindow = timeWindow;
  }

  async waitIfNeeded(estimatedTokens: number = 10000): Promise<void> {
    const now = Date.now();
    
    // 時間窓外の古いリクエストを削除
    this.requests = this.requests.filter(time => now - time < this.timeWindow);
    this.tokenUsage = this.tokenUsage.filter(usage => now - usage.time < this.timeWindow);
    
    // 現在のトークン使用量を計算
    const currentTokens = this.tokenUsage.reduce((sum, usage) => sum + usage.tokens, 0);
    
    // リクエスト数制限チェック
    if (this.requests.length >= this.maxRequests) {
      const oldestRequest = Math.min(...this.requests);
      const waitTime = this.timeWindow - (now - oldestRequest) + 1000;
      
      if (waitTime > 0) {
        console.log(`⏳ リクエスト制限のため ${Math.ceil(waitTime / 1000)} 秒待機します...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
    
    // トークン制限チェック
    if (currentTokens + estimatedTokens > this.maxTokens) {
      const oldestToken = Math.min(...this.tokenUsage.map(u => u.time));
      const waitTime = this.timeWindow - (now - oldestToken) + 1000;
      
      if (waitTime > 0) {
        console.log(`⏳ トークン制限のため ${Math.ceil(waitTime / 1000)} 秒待機します...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
    
    this.requests.push(now);
    this.tokenUsage.push({ time: now, tokens: estimatedTokens });
  }
}

// レート制限インスタンスを作成（より保守的な設定）
const rateLimiter = new TokenRateLimiter(200, 150000, 60000); // 1分間に200リクエスト、150,000トークン

// OpenAIモデル初期化（レート制限対応）
const llm = new ChatOpenAI({
  temperature: 0.1,
  model: "gpt-4o-mini",
  openAIApiKey: process.env.OPENAI_API_KEY,
  maxConcurrency: 1, // 並行リクエスト数を1に制限
  maxRetries: 0, // リトライを無効化（レート制限で制御）
});

console.log('🚀 要約処理を開始します...\n');

// ファイルの存在確認
if (!fs.existsSync(filePath)) {
  console.error(`❌ エラー: ファイルが見つかりません: ${filePath}`);
  process.exit(1);
}

const rawText = fs.readFileSync(filePath, "utf-8");
const text = cleanText(rawText);

// テキストが空でないことを確認
if (!text || text.length === 0) {
  console.error('❌ エラー: 処理可能なテキストが見つかりません');
  process.exit(1);
}

// テキストを文書オブジェクトに変換（トークン制限を考慮してチャンクサイズを調整）
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 30000, // トークン制限を考慮してチャンクサイズを小さく
  chunkOverlap: 1000,
});
const documents = await splitter.createDocuments([text]);

// 進行度ロガーを初期化
const progressLogger = new ProgressLogger(documents.length + 2);

console.log(`📄 ファイル: ${filePath}`);
console.log(`📏 要約の長さ: ${summaryLength}`);
console.log(`📊 処理するテキストの長さ: ${text.length.toLocaleString()}文字`);
console.log(`📑 文書の分割数: ${documents.length}個`);
console.log(`🔢 予想リクエスト数: ${documents.length + 1}回\n`);

progressLogger.logStep('設定の初期化完了');

// カスタムプロンプトテンプレートを作成（より簡潔に）
const mapPrompt = PromptTemplate.fromTemplate(`
${getLengthInstruction(summaryLength)}

テキスト: {text}

要約:`);

const combinePrompt = PromptTemplate.fromTemplate(`
${getLengthInstruction(summaryLength)}

要約リスト:
{text}

最終要約:`);

// 要約チェーンを作成
const chain = await loadSummarizationChain(llm, {
  type: "map_reduce",
  combineMapPrompt: mapPrompt,
  combinePrompt: combinePrompt,
});

progressLogger.logStep('要約チェーンの作成完了');

// 各チャンクを順次処理（トークン制限対応）
console.log('🤖 AIによる要約処理を実行中...');

// 推定トークン数を計算（文字数の約1/4をトークン数として概算）
const estimatedTokensPerChunk = Math.ceil(30000 / 4);

for (let i = 0; i < documents.length; i++) {
  await rateLimiter.waitIfNeeded(estimatedTokensPerChunk);
  console.log(`📝 チャンク ${i + 1}/${documents.length} を処理中...`);
}

// 最終的な要約を実行
await rateLimiter.waitIfNeeded(estimatedTokensPerChunk);
const summary = await chain.invoke({
  input_documents: documents,
});

progressLogger.logStep('要約処理完了');
progressLogger.logComplete();

console.log('=== 要約結果 ===');
console.log(summary.text);
console.log('\n使用方法:');
console.log('node dist/index.js --file=<ファイルパス> --length=<short|medium|long|数値>');
console.log('例: node dist/index.js --file=./sample.txt --length=short');
