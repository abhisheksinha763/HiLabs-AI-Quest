'use client'

import { useState, useEffect } from 'react'

// Backend response interface
interface AnalysisResponse {
  status: string
  message: string
  data_quality_score: number
  num_duplicates: number
  percentage_duplicate: number
  unique_providers: number
  bad_phone_list: Array<{
    license_number: string
    provider_name: string
    phone_number: string
  }>
  provider_file_info: {
    filename: string
    rows: number
    columns: number
  }
  merged_file_info: {
    rows: number
    columns: number
  }
}

// Display data interface
interface DisplayProvider {
  name: string
  specialty: string
  phone: string
  state: string
  license_number?: string
}

// Function to format phone number to +1-xxx-xxx-xxxx
const formatPhoneNumber = (phone: string): string => {
  // Remove all non-digit characters
  const digits = phone.replace(/\D/g, '')
  
  // Check if it's a valid 10-digit US number
  if (digits.length === 10) {
    return `+1-${digits.slice(0, 3)}-${digits.slice(3, 6)}-${digits.slice(6)}`
  }
  
  // Check if it's 11 digits starting with 1
  if (digits.length === 11 && digits.startsWith('1')) {
    const tenDigits = digits.slice(1)
    return `+1-${tenDigits.slice(0, 3)}-${tenDigits.slice(3, 6)}-${tenDigits.slice(6)}`
  }
  
  // Return original if can't format
  return phone
}

// Function to check if phone number is valid
const isValidPhoneNumber = (phone: string): boolean => {
  const digits = phone.replace(/\D/g, '')
  return (digits.length === 10) || (digits.length === 11 && digits.startsWith('1'))
}

interface DisplayData {
  dataQuality: number
  totalProviders: number
  duplicates: number
  duplicatePercentage: number
  invalidPhoneProviders: DisplayProvider[]
}

// Mock data for fallback
const mockData: DisplayData = {
  dataQuality: 86,
  totalProviders: 1024,
  duplicates: 134,
  duplicatePercentage: 12,
  invalidPhoneProviders: [
    { name: "Dr. A. Kumar", specialty: "Cardiology", phone: "12345", state: "NY" },
    { name: "Dr. B. Singh", specialty: "Orthopedics", phone: "9876-AB", state: "CA" },
    { name: "Dr. C. Johnson", specialty: "Neurology", phone: "555-ABCD", state: "TX" },
    { name: "Dr. D. Patel", specialty: "Dermatology", phone: "123", state: "FL" },
    { name: "Dr. E. Williams", specialty: "Pediatrics", phone: "999-999-999X", state: "IL" },
  ]
}

export default function OverviewPage() {
  const [analysisData, setAnalysisData] = useState<AnalysisResponse | null>(null)
  const [displayData, setDisplayData] = useState<DisplayData>(mockData)

  useEffect(() => {
    const storedData = sessionStorage.getItem('analysisResult')
    if (storedData) {
      try {
        const parsed = JSON.parse(storedData) as AnalysisResponse
        setAnalysisData(parsed)
        
        // Transform backend data to display format
        setDisplayData({
          dataQuality: parseFloat((parsed.data_quality_score * 100).toFixed(2)),
          totalProviders: parsed.unique_providers,
          duplicates: parsed.num_duplicates,
          duplicatePercentage: parseFloat(parsed.percentage_duplicate.toFixed(2)),
          invalidPhoneProviders: parsed.bad_phone_list.map(provider => ({
            name: provider.provider_name,
            specialty: "Unknown", // Backend doesn't provide specialty
            phone: provider.phone_number,
            state: "Unknown", // Backend doesn't provide state
            license_number: provider.license_number
          }))
        })
      } catch (error) {
        console.error('Error parsing analysis data:', error)
      }
    }
  }, [])

  const DataQualityGauge = ({ percentage }: { percentage: number }) => {
    const getColor = (pct: number) => {
      if (pct >= 80) return '#27AE60' // Emerald Green
      if (pct >= 60) return '#E67E22' // Amber
      return '#C0392B' // Crimson Red
    }

    const getZoneText = (pct: number) => {
      if (pct >= 80) return 'Healthy'
      if (pct >= 60) return 'Needs Attention'
      return 'Poor'
    }

    const rotation = (percentage / 100) * 180 - 90

    return (
      <div className="flex flex-col items-center mt-12">
        <div className="relative w-48 h-32 mb-4">
          {/* Background arc */}
          <svg className="w-48 h-32" viewBox="0 0 128 80">
            <path
              d="M 10 70 A 54 54 0 0 1 118 70"
              fill="none"
              stroke="#E5E7EB"
              strokeWidth="8"
              strokeLinecap="round"
            />
            {/* Progress arc */}
            <path
              d="M 10 70 A 54 54 0 0 1 118 70"
              fill="none"
              stroke={getColor(percentage)}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${(percentage / 100) * 170} 170`}
            />
          </svg>
          {/* Center dot */}
          <div 
            className="absolute w-3 h-3 bg-gray-800 rounded-full"
            style={{
              top: '87.5%',
              left: '50%',
              transform: 'translate(-50%, -50%)'
            }}
          />
          {/* Needle */}
          <div
            className="absolute w-1 h-8 bg-gray-800"
            style={{ 
              top: '87.5%',
              left: '50%',
              transform: `translate(-50%, -100%) rotate(${rotation}deg)`,
              transformOrigin: 'center bottom'
            }}
          />
        </div>
        <div className="text-3xl font-bold text-[#2C3E50] mb-1">{percentage}%</div>
        <div className="text-sm text-[#7F8C8D]">{getZoneText(percentage)}</div>
      </div>
    )
  }

  const DonutChart = ({ duplicates, total, duplicatePercentage: providedPercentage }: { duplicates: number, total: number, duplicatePercentage?: number }) => {
    // Use provided percentage from backend if available, otherwise calculate
    const duplicatePercentage = providedPercentage !== undefined ? providedPercentage : (duplicates / total) * 100
    const uniquePercentage = 100 - duplicatePercentage
    const circumference = 2 * Math.PI * 30
    const duplicateStroke = (duplicatePercentage / 100) * circumference
    const uniqueStroke = (uniquePercentage / 100) * circumference

    return (
      <div className="flex flex-col items-center">
        <div className="relative w-80 h-80 mb-4">
          <svg className="w-80 h-80 transform -rotate-90" viewBox="0 0 100 100">
            {/* Unique providers (green) */}
            <circle
              cx="50"
              cy="50"
              r="30"
              fill="none"
              stroke="#27AE60"
              strokeWidth="8"
              strokeDasharray={`${uniqueStroke} ${circumference}`}
            />
            {/* Duplicates (red) */}
            <circle
              cx="50"
              cy="50"
              r="30"
              fill="none"
              stroke="#C0392B"
              strokeWidth="8"
              strokeDasharray={`${duplicateStroke} ${circumference}`}
              strokeDashoffset={-uniqueStroke}
            />
          </svg>
          {/* Center label */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-2xl font-bold text-[#2C3E50]">{duplicates}</div>
          </div>
        </div>
        <div className="text-sm text-[#7F8C8D] text-center">
          <div className="flex items-center justify-center space-x-4 mb-2">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-[#C0392B] rounded-full mr-2"></div>
              <span>Duplicates {duplicatePercentage.toFixed(2)}%</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-[#27AE60] rounded-full mr-2"></div>
              <span>Unique {uniquePercentage.toFixed(2)}%</span>
            </div>
          </div>
          <div className="font-medium text-[#2C3E50]">{duplicates} duplicates ({duplicatePercentage.toFixed(2)}%)</div>
        </div>
      </div>
    )
  }

  const SparklineChart = () => {
    const points = [920, 945, 980, 995, 1010, 1015, 1024]
    const max = Math.max(...points)
    const min = Math.min(...points)
    const range = max - min || 1

    return (
      <svg className="w-24 h-8" viewBox="0 0 96 32">
        <polyline
          fill="none"
          stroke="#2E86DE"
          strokeWidth="2"
          points={points.map((point, index) => 
            `${(index * 16)},${32 - ((point - min) / range) * 24}`
          ).join(' ')}
        />
      </svg>
    )
  }

  const InvalidPhoneDropdown = () => {
    const [isExpanded, setIsExpanded] = useState(false)
    const [searchTerm, setSearchTerm] = useState('')

    const filteredProviders = displayData.invalidPhoneProviders.filter(provider =>
      provider.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      provider.specialty.toLowerCase().includes(searchTerm.toLowerCase()) ||
      provider.state.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (provider.license_number && provider.license_number.toLowerCase().includes(searchTerm.toLowerCase()))
    )

    return (
      <div className="bg-white rounded-2xl shadow-lg">
        {/* Dropdown Header */}
        <div 
          className="flex items-center justify-between p-6 cursor-pointer hover:bg-[#F7F9FB] transition-colors"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-[#2C3E50]">Providers with Invalid Phone Numbers</h3>
            <span className="bg-[#C0392B] text-white text-xs px-2 py-1 rounded-full">
              {displayData.invalidPhoneProviders.length}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <svg 
              className={`w-5 h-5 text-[#7F8C8D] transition-transform duration-200 ${
                isExpanded ? 'rotate-180' : ''
              }`} 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>

        {/* Dropdown Content */}
        {isExpanded && (
          <div className="border-t border-gray-200">
            {/* Search Bar */}
            <div className="p-6 pb-4">
              <input
                type="text"
                placeholder="Search providers..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-[#2E86DE] focus:border-transparent"
              />
            </div>

            {/* Table */}
            <div className="px-6 pb-6">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-medium text-[#7F8C8D] text-sm uppercase tracking-wide">Provider Name</th>
                      <th className="text-left py-3 px-4 font-medium text-[#7F8C8D] text-sm uppercase tracking-wide">License Number</th>
                      <th className="text-left py-3 px-4 font-medium text-[#7F8C8D] text-sm uppercase tracking-wide">Invalid Phone</th>
                      <th className="text-left py-3 px-4 font-medium text-[#7F8C8D] text-sm uppercase tracking-wide">Valid Phone</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredProviders.map((provider, index) => (
                      <tr 
                        key={index} 
                        className={`border-b border-gray-100 hover:bg-[#F7F9FB] cursor-pointer transition-colors ${
                          index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                        }`}
                        onClick={() => console.log(`Open provider detail for ${provider.name}`)}
                      >
                        <td className="py-4 px-4 font-medium text-[#2C3E50]">{provider.name}</td>
                        <td className="py-4 px-4 text-[#7F8C8D] font-mono">{provider.license_number || 'N/A'}</td>
                        <td className="py-4 px-4">
                          <div className="flex items-center space-x-2">
                            <svg className="w-4 h-4 text-[#C0392B]" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                            <span className="text-[#C0392B] font-mono">{provider.phone}</span>
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          {isValidPhoneNumber(provider.phone) ? (
                            <span className="text-green-600 font-mono font-medium">
                              {formatPhoneNumber(provider.phone)}
                            </span>
                          ) : (
                            <span className="text-gray-400 font-mono">
                              Invalid format
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {filteredProviders.length === 0 && searchTerm && (
                <div className="text-center py-8">
                  <div className="text-[#7F8C8D]">No providers found matching "{searchTerm}"</div>
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200">
                <div className="text-sm text-[#7F8C8D]">
                  Showing {filteredProviders.length} of {displayData.invalidPhoneProviders.length} providers
                </div>
                <div className="flex items-center space-x-2">
                  <button className="px-3 py-1 text-sm text-[#7F8C8D] hover:text-[#2C3E50] disabled:opacity-50" disabled>
                    Previous
                  </button>
                  <span className="px-3 py-1 bg-[#2E86DE] text-white rounded text-sm">1</span>
                  <button className="px-3 py-1 text-sm text-[#7F8C8D] hover:text-[#2C3E50]">
                    Next
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#F7F9FB] p-6">
      {/* KPI Ribbon */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* Data Quality Card */}
        <div 
          className="bg-white rounded-2xl shadow-lg p-6 cursor-pointer hover:shadow-xl transition-shadow"
          onClick={() => console.log('Navigate to data quality breakdown')}
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-[#7F8C8D] uppercase tracking-wide">Data Quality</h3>
            <div className="w-4 h-4 bg-[#2E86DE] rounded-full flex items-center justify-center cursor-help" title="Completeness, Conformance, Consistency, Dedup Purity">
              <span className="text-white text-xs font-bold">i</span>
            </div>
          </div>
          <DataQualityGauge percentage={displayData.dataQuality} />
        </div>

        {/* Total Duplicates Card */}
        <div 
          className="bg-white rounded-2xl shadow-lg p-6 cursor-pointer hover:shadow-xl transition-shadow"
          onClick={() => console.log('Filter to show duplicates only')}
        >
          <h3 className="text-sm font-medium text-[#7F8C8D] uppercase tracking-wide mb-4">Total Duplicates</h3>
          <DonutChart duplicates={displayData.duplicates} total={displayData.totalProviders} duplicatePercentage={displayData.duplicatePercentage} />
        </div>

        {/* Genuine Providers Card */}
        <div 
          className="bg-white rounded-2xl shadow-lg p-6 cursor-pointer hover:shadow-xl transition-shadow flex flex-col justify-center"
          onClick={() => console.log('Navigate to explore tab with validated providers')}
        >
          <h3 className="text-sm font-medium text-[#7F8C8D] uppercase tracking-wide mb-4 text-center">Genuine Providers</h3>
          <div className="flex flex-col items-center justify-center flex-grow">
            <div className="text-4xl font-bold text-[#2C3E50] mb-2">
              {(Math.round((displayData.duplicates * 100) / displayData.duplicatePercentage) - displayData.duplicates).toLocaleString()}
            </div>
            <div className="text-sm text-[#7F8C8D]">providers validated and unique</div>
          </div>
        </div>
      </div>

      {/* Error List Section - Collapsible Dropdown */}
      <InvalidPhoneDropdown />

    </div>
  )
}
