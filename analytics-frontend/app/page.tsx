'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

const US_STATES = [
  'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
  'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
  'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
  'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico',
  'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
  'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
  'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

export default function Home() {
  const router = useRouter()
  const [uploadedFiles, setUploadedFiles] = useState({
    uncleaned: null as File | null,
    stateWise: [] as Array<{state: string, file: File}>,
    npiBased: null as File | null,
  })
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<string>('')

  const handleFileUpload = (type: 'uncleaned' | 'npiBased', file: File) => {
    setUploadedFiles(prev => ({
      ...prev,
      [type]: file
    }))
  }

  const handleStateWiseUpload = (state: string, file: File) => {
    setUploadedFiles(prev => ({
      ...prev,
      stateWise: [...prev.stateWise, { state, file }]
    }))
  }

  const removeStateWiseFile = (index: number) => {
    setUploadedFiles(prev => ({
      ...prev,
      stateWise: prev.stateWise.filter((_, i) => i !== index)
    }))
  }

  const handleAnalyze = async () => {
    const allFiles = []
    if (uploadedFiles.uncleaned) allFiles.push(uploadedFiles.uncleaned)
    if (uploadedFiles.npiBased) allFiles.push(uploadedFiles.npiBased)
    uploadedFiles.stateWise.forEach(item => allFiles.push(item.file))
    
    if (allFiles.length === 0) {
      alert('Please upload at least one file before analyzing.')
      return
    }

    setIsAnalyzing(true)
    setAnalysisResult('')

    try {
      const formData = new FormData()
      
      // Debug: Log all files being sent
      console.log('🔍 DEBUGGING: Files being sent to backend (in correct order):')
      console.log('='.repeat(50))
      
      // 1. First: Upload uncleaned data
      if (uploadedFiles.uncleaned) {
        console.log(`1. ✓ Uncleaned file: ${uploadedFiles.uncleaned.name} (${uploadedFiles.uncleaned.size} bytes)`)
        formData.append('files', uploadedFiles.uncleaned)
      } else {
        console.log('1. ✗ No uncleaned file')
      }
      
      // 2. Second: Upload statewise data in order of their upload
      if (uploadedFiles.stateWise.length > 0) {
        uploadedFiles.stateWise.forEach((item, index) => {
          console.log(`${2 + index}. ✓ State file: ${item.file.name} for ${item.state} (${item.file.size} bytes)`)
          formData.append('files', item.file)
        })
      } else {
        console.log('2. ✗ No state-wise files')
      }
      
      // 3. Last: Upload NPI based data
      const npiOrder = 2 + uploadedFiles.stateWise.length
      if (uploadedFiles.npiBased) {
        console.log(`${npiOrder}. ✓ NPI file: ${uploadedFiles.npiBased.name} (${uploadedFiles.npiBased.size} bytes)`)
        formData.append('files', uploadedFiles.npiBased)
      } else {
        console.log(`${npiOrder}. ✗ No NPI file`)
      }

      const totalFiles = (uploadedFiles.uncleaned ? 1 : 0) + (uploadedFiles.npiBased ? 1 : 0) + uploadedFiles.stateWise.length
      console.log(`📊 Total files: ${totalFiles}`)
      console.log('🌐 API URL:', `${process.env.NEXT_PUBLIC_API_URL}/process`)
      console.log('='.repeat(50))
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/process`, {
        method: 'POST',
        headers: {
          'accept': 'application/json',
        },
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      // Debug: Log raw response first
      console.log('🔍 RAW BACKEND RESPONSE:')
      console.log('='.repeat(60))
      console.log('Raw data_quality_score:', data.data_quality_score)
      console.log('Raw response:', JSON.stringify(data, null, 2))
      console.log('='.repeat(60))
      
      // Log detailed results to terminal
      console.log('🔍 ANALYSIS RESULTS')
      console.log('='.repeat(60))
      console.log('Status:', data.status)
      console.log('Message:', data.message)
      console.log('Data Quality Score:', `${(data.data_quality_score * 100).toFixed(1)}%`)
      console.log('Total Unique Providers:', data.unique_providers.toLocaleString())
      console.log('Duplicates Found:', data.num_duplicates.toLocaleString())
      console.log('Duplicate Percentage:', `${data.percentage_duplicate.toFixed(1)}%`)
      console.log('Invalid Phone Numbers:', data.bad_phone_list.length)
      
      if (data.provider_file_info) {
        console.log('\n📊 Provider File Info:')
        console.log('  Filename:', data.provider_file_info.filename)
        console.log('  Rows:', data.provider_file_info.rows.toLocaleString())
        console.log('  Columns:', data.provider_file_info.columns)
      }
      
      if (data.merged_file_info) {
        console.log('\n🔗 Merged File Info:')
        console.log('  Total Rows:', data.merged_file_info.rows.toLocaleString())
        console.log('  Total Columns:', data.merged_file_info.columns)
      }
      
      if (data.bad_phone_list && data.bad_phone_list.length > 0) {
        console.log('\n📞 Sample Invalid Phone Numbers:')
        data.bad_phone_list.slice(0, 5).forEach((provider, index) => {
          console.log(`  ${index + 1}. ${provider.provider_name} (${provider.license_number}): ${provider.phone_number}`)
        })
        if (data.bad_phone_list.length > 5) {
          console.log(`  ... and ${data.bad_phone_list.length - 5} more`)
        }
      }
      
      console.log('='.repeat(60))
      
      // Store analysis data and redirect to dashboard
      sessionStorage.setItem('analysisResult', JSON.stringify(data))
      router.push('/home/dashboard/overview')
    } catch (error: any) {
      console.error('Analysis failed:', error)
      setAnalysisResult(`Error: ${error.message}`)
      setIsAnalyzing(false)
    }
  }

  const UploadButton = ({ 
    title, 
    type, 
    description 
  }: { 
    title: string
    type: 'uncleaned' | 'npiBased'
    description: string
  }) => {
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (file) {
        handleFileUpload(type, file)
      }
    }

    return (
      <div className="bg-white rounded-lg shadow-md p-6 border-2 border-dashed border-gray-300 hover:border-primary-500 transition-colors h-full flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
        <p className="text-sm text-gray-600 mb-4 flex-grow">{description}</p>
        
        <div className="mt-auto">
          <label className="block">
            <input
              type="file"
              onChange={handleFileChange}
              className="hidden"
              accept=".csv,.xlsx,.xls,.json"
            />
            <div className="cursor-pointer bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-md text-center transition-colors">
              {uploadedFiles[type] ? 'File Selected ✓' : 'Choose File'}
            </div>
          </label>
          
          {uploadedFiles[type] && (
            <p className="text-xs text-green-600 mt-2 truncate">
              Selected: {uploadedFiles[type]?.name}
            </p>
          )}
        </div>
      </div>
    )
  }

  const StateWiseUpload = () => {
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files
      if (files) {
        Array.from(files).forEach((file, index) => {
          // Use filename or index as state identifier for multiple files
          const stateIdentifier = `File ${index + 1}`
          handleStateWiseUpload(stateIdentifier, file)
        })
      }
    }

    return (
      <div className="bg-white rounded-lg shadow-md p-6 border-2 border-dashed border-gray-300 hover:border-primary-500 transition-colors h-full flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload State Wise Data</h3>
        <p className="text-sm text-gray-600 mb-4 flex-grow">Data organized and categorized by geographical states (multiple files supported)</p>
        
        <div className="mt-auto space-y-3">
          <label className="block">
            <input
              type="file"
              onChange={handleFileChange}
              className="hidden"
              accept=".csv,.xlsx,.xls,.json"
              multiple
            />
            <div className="cursor-pointer bg-primary-500 hover:bg-primary-600 text-white px-4 py-2 rounded-md text-center transition-colors">
              Choose Files (Multiple)
            </div>
          </label>
          
          {uploadedFiles.stateWise.length > 0 && (
            <div className="max-h-32 overflow-y-auto">
              <p className="text-xs font-medium text-gray-700 mb-1">Uploaded files:</p>
              {uploadedFiles.stateWise.map((item, index) => (
                <div key={index} className="flex items-center justify-between text-xs text-green-600 mb-1">
                  <span className="truncate">{item.state}: {item.file.name}</span>
                  <button
                    onClick={() => removeStateWiseFile(index)}
                    className="ml-2 text-red-500 hover:text-red-700"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Data Analytics Platform
          </h1>
          <p className="text-lg text-gray-600">
            Upload your data files and run comprehensive analysis
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <UploadButton
            title="Upload Uncleaned Data"
            type="uncleaned"
            description="Raw data that needs preprocessing and cleaning before analysis"
          />
          
          <StateWiseUpload />
          
          <UploadButton
            title="Upload NPI Based Data"
            type="npiBased"
            description="National Provider Identifier based healthcare data"
          />
        </div>

        <div className="text-center">
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className={`px-8 py-4 rounded-lg text-xl font-semibold shadow-lg transition-colors transform hover:scale-105 ${
              isAnalyzing 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyse'}
          </button>
          
          <div className="mt-4 text-sm text-gray-600">
            {(uploadedFiles.uncleaned ? 1 : 0) + (uploadedFiles.npiBased ? 1 : 0) + uploadedFiles.stateWise.length} files uploaded
          </div>
        </div>

        {/* Analysis Loading Modal */}
        {isAnalyzing && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-8 max-w-md w-mx-4 text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-green-600 mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Analyzing Data</h3>
              <p className="text-gray-600 mb-4">
                Processing {(uploadedFiles.uncleaned ? 1 : 0) + (uploadedFiles.npiBased ? 1 : 0) + uploadedFiles.stateWise.length} files...
              </p>
              <div className="text-sm text-gray-500">
                This may take a few moments
              </div>
            </div>
          </div>
        )}

        <div className="mt-12 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload Status</h2>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-700">Uncleaned Data:</span>
              <span className={uploadedFiles.uncleaned ? "text-green-600" : "text-gray-400"}>
                {uploadedFiles.uncleaned ? "✓ Uploaded" : "Not uploaded"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">State Wise Data:</span>
              <span className={uploadedFiles.stateWise.length > 0 ? "text-green-600" : "text-gray-400"}>
                {uploadedFiles.stateWise.length > 0 ? `✓ ${uploadedFiles.stateWise.length} files uploaded` : "Not uploaded"}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-700">NPI Based Data:</span>
              <span className={uploadedFiles.npiBased ? "text-green-600" : "text-gray-400"}>
                {uploadedFiles.npiBased ? "✓ Uploaded" : "Not uploaded"}
              </span>
            </div>
          </div>
        </div>

        {analysisResult && (
          <div className="mt-6 bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h2>
            <pre className="bg-gray-100 p-4 rounded-md text-sm overflow-auto whitespace-pre-wrap">
              {analysisResult}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
